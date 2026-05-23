"""LangGraph-based navigation agent for indoor robot navigation (FSR-VLN).

Architecture follows refinement.md: three LLM conversational nodes, with VLM used
only inside the scene-graph query pipeline for slow visual re-ranking (unchanged).

Graph topology (one invocation per user message):

    START
      │
      ▼
    router ─────────────────────────────────────────────
      │ IDLE              │ SUGGEST      │ NAV_SELECT
      ▼                   ▼              ▼
    intent_node    suggest_classify  nav_selector ──► END
      │                   │
      │ explicit           │ choice/ambiguous ──► search_node ──► END
      │                   │
      │ implicit           │ new_intent ──► intent_node (re-enter)
      │
      ▼
    suggest_emit ──► END   (sets phase=SUGGEST, returns suggestion)

    search_node outcomes:
      same_room:  navigate directly (action set, phase=IDLE)
      not_found:  ask to retry     (phase=IDLE)
      multi_room: present list     (phase=NAV_SELECT)

Node responsibilities:
  router           — passthrough; conditional edges dispatch to correct handler
  intent_node      — LLM: classify explicit/implicit, extract {object, room?, floor?}
                     grounded to available label lists; episodic history context
  suggest_emit     — LLM: generate one natural clarification question from alternatives
  suggest_classify — LLM: classify user reply (explicit_choice|ambiguous|new_intent)
  search_node      — query scene graph; resolve same_room/not_found/multi_room
  nav_selector     — LLM: multi-turn disambiguation among candidate locations;
                     candidate list re-injected each turn; episodic history context

Design constraints (per refinement.md):
  - All output labels are grounded to provided label lists — no hallucination.
  - SuggestNode fires at most once per goal attempt.
  - Episodic message history is shared across all nodes.
  - VLM re-ranking inside _query_scene_graph is controlled by slow_reasoning (unchanged).
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

_log = logging.getLogger(__name__)

# Coordinate transform: scene-graph frame → LiDAR-map frame
_T_TO_MAP = np.linalg.inv(
    np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
)

# Dialogue phase labels
PHASE_IDLE = "IDLE"
PHASE_SUGGEST = "SUGGEST"
PHASE_NAV_SELECT = "NAV_SELECT"

_AFFIRMATIVES = frozenset(
    {
        "yes",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "please",
        "go",
        "do it",
        "confirm",
        "y",
        "any",
        "any of them",
        "either",
        "whatever",
        "doesn't matter",
        "dont care",
        "don't care",
    }
)
_NEGATIVES = frozenset(
    {
        "no",
        "nope",
        "nah",
        "not that",
        "never mind",
        "cancel",
        "stop",
    }
)

FALLBACK = "I am a navigation assistant. How can I help you?"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class NavAgentState(TypedDict, total=False):
    """Shared state threaded through every node in the navigation graph."""

    # --- caller-supplied per turn ---
    user_message: str
    llm_enable: bool  # enable LLM for query understanding / goal retrieval selection
    slow_reasoning: bool  # enable VLM slow-reasoning re-ranking (independent of llm_enable)
    room_labels: List[str]  # available room types from config
    object_labels: List[str]  # available object labels from config/pipeline
    floor_labels: List[str]  # available floor names from scene graph
    scene_graph: Any  # loaded Graph instance

    # --- dialogue state (persisted across turns by NavigationAgent) ---
    phase: str  # one of the PHASE_* constants
    goal: Dict[str, Any]  # {object, room?, floor?} from IntentNode
    alternatives: List[str]  # implicit intent: ordered alternative labels
    candidates: List[Dict[str, Any]]  # scene-graph hit list (multi_room outcome)
    messages: List[Dict[str, str]]  # episodic conversation history (all turns)

    # --- per-turn outputs ---
    response: str
    action: Optional[Dict[str, Any]]  # {name, x, y, z} or None


# ---------------------------------------------------------------------------
# Pure utility functions (shared by nodes)
# ---------------------------------------------------------------------------


def _is_affirmative(msg: str) -> bool:
    return any(a in msg.strip().lower() for a in _AFFIRMATIVES)


def _is_negative(msg: str) -> bool:
    return any(n in msg.strip().lower() for n in _NEGATIVES)


def _navigate_to(candidate: Dict) -> tuple:
    response = (
        f"I will navigate you to the {candidate['name']} "
        f"in {candidate['room']}, {candidate['floor']}."
    )
    action = {
        "name": candidate["name"],
        "x": candidate["x"],
        "y": candidate["y"],
        "z": candidate["z"],
    }
    return response, action


def _format_candidates(obj: str, candidates: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(candidates):
        oid = c.get("object_id", "")
        score = c.get("score")
        score_str = f" [{score:.4f}]" if score is not None else ""
        room_id = c.get("room_id", "")
        room_str = f"{c['room']} ({room_id})" if room_id and room_id != c["room"] else c["room"]
        lines.append(f"  {i + 1}. {c['name']} ({oid}){score_str} \u2014 {room_str}, {c['floor']}")
    plural = f"{obj}s" if not obj.endswith("s") else obj
    return f"Found {len(candidates)} {plural}:\n" + "\n".join(lines) + "\nWhere do you want to go?"


def _query_scene_graph(
    scene_graph: Any,
    queries: List[str],
    top_k: int = 5,
    slow_reasoning: bool = True,
    llm_enable: bool = True,
) -> List[Dict]:
    """Query scene graph with multiple object queries → deduplicated candidate list.

    VLM re-ranking is controlled by the slow_reasoning flag and is not modified here.
    This function is intentionally kept identical to the original implementation.
    """
    if scene_graph is None:
        return []
    seen: set = set()
    results: List[Dict] = []
    for query in queries:
        _log.debug(
            "scene-graph query: %r  top_k=%d  slow_reasoning=%s", query, top_k, slow_reasoning
        )
        try:
            _, rooms, objects, res_dict = scene_graph.query_hierarchy(
                query, top_k=top_k, slow_reasoning=slow_reasoning, llm_enable=llm_enable
            )
            scores = (res_dict or {}).get("object_scores", [])
        except Exception as exc:
            _log.error("Scene graph query failed for %r: %s", query, exc)
            continue
        for i, (obj, room) in enumerate(zip(objects, rooms)):
            oid = obj.object_id
            if oid in seen:
                continue
            seen.add(oid)
            center = np.array(obj.pcd.get_center())
            c_map = (_T_TO_MAP @ np.hstack((center, 1.0)))[:3]
            parts = str(oid).split("_")
            floor_label = "?"
            if parts and parts[0].isdigit() and scene_graph.floors:
                idx = int(parts[0])
                if idx < len(scene_graph.floors):
                    fl = scene_graph.floors[idx]
                    floor_label = fl.name or f"Floor {idx}"
            results.append(
                {
                    "name": obj.name or f"object_{oid}",
                    "query": query,
                    "room": room.name or room.room_id,
                    "room_id": room.room_id,
                    "floor": floor_label,
                    "score": float(scores[i]) if i < len(scores) else None,
                    "x": float(c_map[0]),
                    "y": float(c_map[1]),
                    "z": float(c_map[2]),
                    "object_id": oid,
                }
            )
    return results


# ---------------------------------------------------------------------------
# LLM call helpers
# ---------------------------------------------------------------------------


def _llm_call(system: str, history: List[Dict], user_message: str) -> str:
    """Single LLM chat completion: system prompt + optional history + user message.

    Uses the module-level cached client and send_query_cached for deduplication.
    Returns the raw response content string.
    """
    from memory.hmsg.tools.llm import _get_cached_client, send_query_cached

    _, model = _get_cached_client()
    full_messages = (
        [{"role": "system", "content": system}]
        + list(history)
        + [{"role": "user", "content": user_message}]
    )
    return send_query_cached(model, json.dumps(full_messages, ensure_ascii=False), temperature=0.0)


def _parse_json_llm(raw: str) -> Dict:
    """Strip markdown fences and parse the first JSON object from an LLM response."""
    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw.strip())
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in LLM output: {raw!r}")
    return json.loads(m.group())


# ---------------------------------------------------------------------------
# Non-LLM fallbacks
# ---------------------------------------------------------------------------


def _suggest_template(alternatives: List[str]) -> str:
    """Template-based clarification message when LLM is disabled or unavailable."""
    if not alternatives:
        return FALLBACK
    primary = alternatives[0]
    article = "an" if primary[:1].lower() in "aeiou" else "a"
    if len(alternatives) > 1:
        others = " and ".join(alternatives[1:3])
        return f"Did you mean {article} {primary}? I also see {others}."
    return f"Do you want to go to {article} {primary}?"


def _suggest_classify_fallback(user_msg: str, alternatives: List[str]) -> Dict:
    """Keyword-based reply classification when LLM is disabled."""
    lower = user_msg.lower()
    if _is_negative(lower):
        return {"type": "new_intent"}
    for alt in alternatives:
        if alt.lower() in lower:
            return {"type": "explicit_choice", "object": alt}
    return {"type": "ambiguous"}


def _nav_select_fallback(user_msg: str, candidates: List[Dict], obj: str) -> Dict:
    """Location keyword matching for candidate disambiguation when LLM is disabled."""
    lower = user_msg.lower()
    if _is_affirmative(lower):
        return {"action": "navigate", "index": 0}
    if _is_negative(lower) or any(
        w in lower for w in ("start over", "restart", "something else", "never mind")
    ):
        return {"action": "restart"}
    matched = [
        i
        for i, c in enumerate(candidates)
        if (c.get("room") or "").lower() in lower
        or (c.get("floor") or "").lower() in lower
        or (c.get("name") or "").lower() in lower
    ]
    if not matched:
        matched = [i for i in range(len(candidates)) if str(i + 1) in lower]
    if len(matched) == 1:
        return {"action": "navigate", "index": matched[0]}
    if len(matched) > 1:
        sub = [candidates[i] for i in matched]
        return {"action": "clarify", "message": _format_candidates(obj, sub)}
    return {"action": "clarify", "message": _format_candidates(obj, candidates)}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


def router_node(state: NavAgentState) -> dict:
    """Passthrough — conditional edges handle routing based on phase."""
    return {}


def _extract_intent(
    user_msg: str,
    object_labels: List[str],
    room_labels: List[str],
    floor_labels: List[str],
    history: List[Dict],
    llm_enable: bool = True,
    slow_reasoning: bool = False,
) -> Dict:
    """Extract navigation intent from a user message, independent of graph state.

    Returns a dict with one of the following shapes:
      Explicit: {"intent": "explicit", "object": <label>, "room": <label|None>, "floor": <label|None>}
      Implicit: {"intent": "implicit", "alternatives": [<label>, ...]}

    When llm_enable is False or the LLM call fails, attempts keyword matching.
    If keyword matching finds a match, returns explicit intent.
    If no match found:
      - slow_reasoning=True → returns explicit with full user_msg for CLIP semantic search
      - slow_reasoning=False → returns implicit with alternatives for clarification
    This function does NOT touch the scene graph.
    """
    if llm_enable and object_labels:
        from memory.hmsg.tools.prompts import make_intent_node_system

        system = make_intent_node_system(object_labels, room_labels, floor_labels)
        try:
            raw = _llm_call(system, history, user_msg)
            data = _parse_json_llm(raw)
            _log.debug("_extract_intent LLM: %r \u2192 %s", user_msg, data)

            # Guard: if the LLM says explicit but the object label does not literally
            # appear in the user message, the LLM inferred it indirectly → downgrade to
            # implicit so the agent confirms with the user first.
            if data.get("intent") == "explicit" and data.get("object"):
                obj_label = data["object"].lower()
                msg_lower = user_msg.lower()
                if obj_label not in msg_lower:
                    alts = [data["object"]]
                    _log.debug(
                        "_extract_intent: LLM inferred %r from indirect message %r → implicit",
                        data["object"],
                        user_msg,
                    )
                    return {"intent": "implicit", "alternatives": alts}
            return data
        except Exception as exc:
            _log.warning("_extract_intent LLM failed (%s); falling back to keyword matching", exc)

    # If no label inventory is available, use QueryParser for unconstrained LLM
    # extraction so goal.object is a clean label ("bed") not the raw sentence.
    if not object_labels:
        if llm_enable:
            try:
                from memory.hmsg.tools.llm import QueryParser

                _parser = QueryParser()
                floor_h, room_h, obj_h, alts = _parser.parse_with_alts(user_msg)
                if alts:
                    # LLM returned alternatives → query is indirect/implicit.
                    # Put the primary object first so it appears as the top suggestion.
                    all_alts = ([obj_h] + alts) if obj_h else alts
                    _log.debug("_extract_intent unconstrained LLM → implicit alts=%s", all_alts)
                    return {"intent": "implicit", "alternatives": all_alts}
                if obj_h:
                    # Guard: if the extracted label doesn't literally appear in the
                    # user message the intent was inferred (e.g. "I want to sleep" → "bed").
                    # Downgrade to implicit so the agent confirms before navigating.
                    if obj_h.lower() not in user_msg.lower():
                        _log.debug(
                            "_extract_intent: inferred %r from %r → implicit",
                            obj_h,
                            user_msg,
                        )
                        return {"intent": "implicit", "alternatives": [obj_h]}
                    _log.debug("_extract_intent unconstrained LLM → explicit obj=%r", obj_h)
                    return {
                        "intent": "explicit",
                        "object": obj_h,
                        "room": room_h,
                        "floor": floor_h,
                    }
            except Exception as exc:
                _log.warning(
                    "_extract_intent unconstrained LLM failed (%s); using raw message", exc
                )
        _log.debug("_extract_intent no object_labels; using raw user message: %r", user_msg)
        return {"intent": "explicit", "object": user_msg, "room": None, "floor": None}

    # Keyword matching fallback: try to find object labels in user message
    if object_labels:
        msg_lower = user_msg.lower()
        for label in object_labels:
            if label.lower() in msg_lower:
                _log.debug("_extract_intent keyword match: %r found in %r", label, user_msg)
                return {"intent": "explicit", "object": label, "room": None, "floor": None}

        # No explicit match found
        if slow_reasoning:
            # With VLM, pass the full user message for CLIP semantic matching
            _log.debug(
                "_extract_intent slow_reasoning=True; querying with full message: %r", user_msg
            )
            return {"intent": "explicit", "object": user_msg, "room": None, "floor": None}
        else:
            # Ask user to clarify among alternatives
            _log.debug("_extract_intent no keyword match; returning implicit with alternatives")
            return {"intent": "implicit", "alternatives": object_labels}

    return {"intent": "explicit", "object": None}


def intent_node(state: NavAgentState) -> dict:
    """IntentNode: classify explicit/implicit, extract navigation goal.

    Delegates extraction to _extract_intent() then formats the state update.
    All output labels are grounded to the provided label lists.
    """
    user_msg = state["user_message"]
    history = list(state.get("messages") or [])

    data = _extract_intent(
        user_msg,
        state.get("object_labels", []),
        state.get("room_labels", []),
        state.get("floor_labels", []),
        history,
        llm_enable=bool(state.get("llm_enable", True)),
        slow_reasoning=bool(state.get("slow_reasoning", False)),
    )

    updated_messages = history + [{"role": "user", "content": user_msg}]

    if data.get("intent") == "implicit":
        alternatives = [str(a) for a in data.get("alternatives", []) if a]
        if not alternatives:
            return {
                "messages": updated_messages,
                "phase": PHASE_IDLE,
                "response": FALLBACK,
                "action": None,
            }
        return {
            "messages": updated_messages,
            "alternatives": alternatives,
            "goal": {},
            "phase": PHASE_SUGGEST,
        }

    # explicit (or fallback)
    obj = data.get("object")
    if not obj:
        return {
            "messages": updated_messages,
            "phase": PHASE_IDLE,
            "response": FALLBACK,
            "action": None,
        }
    return {
        "messages": updated_messages,
        "goal": {"object": obj, "room": data.get("room"), "floor": data.get("floor")},
        "alternatives": [],
        "phase": PHASE_IDLE,
    }


def suggest_emit_node(state: NavAgentState) -> dict:
    """SuggestNode — emit phase: generate clarification question.

    Called when IntentNode returns implicit. Sets phase=SUGGEST so the next
    turn routes to suggest_classify_node. Only one suggestion turn is allowed
    per goal attempt (per refinement.md constraint).

    To avoid an extra latency-sensitive LLM round trip on every implicit turn,
    this node uses a deterministic template by default. Optional LLM phrasing
    is enabled only when both llm_enable and slow_reasoning are true.
    """
    alternatives = list(state.get("alternatives") or [])
    history = list(state.get("messages") or [])

    llm_enable_emit = bool(
        state.get("llm_enable", True) and state.get("slow_reasoning", False) and alternatives
    )

    if llm_enable_emit:
        from memory.hmsg.tools.prompts import make_suggest_emit_system

        system = make_suggest_emit_system(alternatives)
        try:
            raw = _llm_call(system, [], "Generate the suggestion.")
            data = _parse_json_llm(raw)
            message = data.get("message") or _suggest_template(alternatives)
        except Exception as exc:
            _log.debug("SuggestEmit LLM failed (%s); using template", exc)
            message = _suggest_template(alternatives)
    else:
        message = _suggest_template(alternatives)

    return {
        "messages": history + [{"role": "assistant", "content": message}],
        "phase": PHASE_SUGGEST,
        "response": message,
        "action": None,
    }


def suggest_classify_node(state: NavAgentState) -> dict:
    """SuggestNode — classify phase (LLM): classify user reply to the suggestion.

    Outcomes:
      explicit_choice — user named one alternative → set goal, route to search
      ambiguous       — user agreed/unclear → take alternatives[0], route to search
      new_intent      — user has a different goal → route back to intent_node

    Per refinement.md: only one suggestion turn; ambiguous defaults to alternatives[0].
    For new_intent: goal and alternatives are cleared so intent_node re-extracts
    from the current user_message.
    """
    user_msg = state["user_message"]
    alternatives = list(state.get("alternatives") or [])
    history = list(state.get("messages") or [])

    if state.get("llm_enable", True) and alternatives:
        from memory.hmsg.tools.prompts import make_suggest_classify_system

        system = make_suggest_classify_system(alternatives)
        try:
            raw = _llm_call(system, history, user_msg)
            data = _parse_json_llm(raw)
            _log.debug("SuggestClassify LLM: %r → %s", user_msg, data)
        except Exception as exc:
            _log.warning("SuggestClassify LLM failed (%s); using keyword fallback", exc)
            data = _suggest_classify_fallback(user_msg, alternatives)
    else:
        data = _suggest_classify_fallback(user_msg, alternatives)

    reply_type = data.get("type", "ambiguous")

    if reply_type == "new_intent":
        # Do NOT update messages — intent_node will append user_message when it runs,
        # preserving correct episodic order without duplication.
        return {
            "goal": {},
            "alternatives": [],
            "phase": PHASE_IDLE,
        }

    # explicit_choice or ambiguous → resolve to a single object label
    obj = data.get("object") if reply_type == "explicit_choice" else None
    if not obj and alternatives:
        obj = alternatives[0]
    if not obj:
        return {
            "messages": history + [{"role": "user", "content": user_msg}],
            "phase": PHASE_IDLE,
            "response": FALLBACK,
            "action": None,
        }
    return {
        "messages": history + [{"role": "user", "content": user_msg}],
        "goal": {"object": obj, "room": None, "floor": None},
        "alternatives": [],
        "phase": PHASE_IDLE,
    }


def search_node(state: NavAgentState) -> dict:
    """Query the scene graph with the resolved goal object.

    VLM re-ranking is controlled by ``slow_reasoning`` in state (independent of ``llm_enable``).

    Outcomes:
      same_room  — single location → navigate directly (action set, phase=IDLE)
      not_found  — no match       → ask to retry     (phase=IDLE)
      multi_room — multiple locs  → present list      (phase=NAV_SELECT)
    """
    goal = state.get("goal") or {}
    obj = goal.get("object", "")
    if not obj:
        return {"phase": PHASE_IDLE, "response": FALLBACK, "action": None}

    # Query using the resolved goal object. When intent fallback passes the raw
    # user message as the goal object, this still preserves free-form semantics.
    query = obj

    candidates = _query_scene_graph(
        state.get("scene_graph"),
        [query],
        slow_reasoning=bool(state.get("slow_reasoning", False)),
        # Intent was already extracted by intent_node / suggest_classify_node;
        # suppress the redundant parse_hierarchy_query call inside query_hierarchy.
        llm_enable=False,
    )
    messages = list(state.get("messages") or [])

    if not candidates:
        # Use the query string (extracted object label) not the raw goal sentence.
        display_name = query if query != obj else obj
        msg = (
            f"I couldn't find a {display_name} in the scene. "
            "Could you describe what you're looking for?"
        )
        return {
            "messages": messages + [{"role": "assistant", "content": msg}],
            "phase": PHASE_IDLE,
            "candidates": [],
            "response": msg,
            "action": None,
        }

    # Single candidate or llm disabled → auto-navigate to top scorer.
    # Multiple candidates with llm enabled → present list for user selection.
    llm_enable = bool(state.get("llm_enable", True))
    if len(candidates) == 1 or not llm_enable:
        resp, action = _navigate_to(candidates[0])
        return {
            "messages": messages + [{"role": "assistant", "content": resp}],
            "phase": PHASE_IDLE,
            "candidates": [],
            "response": resp,
            "action": action,
        }

    # multi_candidates + llm_enable: present candidate list, enter NAV_SELECT phase
    resp = _format_candidates(obj, candidates)
    return {
        "messages": messages + [{"role": "assistant", "content": resp}],
        "phase": PHASE_NAV_SELECT,
        "candidates": candidates,
        "response": resp,
        "action": None,
        "llm_enable": state.get("llm_enable", True),
    }


def nav_selector_node(state: NavAgentState) -> dict:
    """NavSelectorNode (LLM, multi-turn): disambiguate multi-room candidates.

    System prompt context (per refinement.md):
      - Full candidate list (re-injected each turn)
      - Full episodic message history (for reference resolution)

    Outcomes:
      navigate — user selected a location → navigate (phase=IDLE)
      clarify  — still ambiguous         → ask one more question (phase=NAV_SELECT)
      restart  — user abandons goal      → clear state, phase=IDLE
    """
    user_msg = state["user_message"]
    candidates = list(state.get("candidates") or [])
    history = list(state.get("messages") or [])
    obj = (state.get("goal") or {}).get("object", "target")

    updated_messages = history + [{"role": "user", "content": user_msg}]

    if not candidates:
        return {
            "messages": updated_messages,
            "phase": PHASE_IDLE,
            "response": FALLBACK,
            "action": None,
        }

    if state.get("llm_enable", True):
        from memory.hmsg.tools.prompts import make_nav_selector_system

        system = make_nav_selector_system(candidates)
        try:
            # Full episodic history injected so the LLM can resolve "the other one" etc.
            raw = _llm_call(system, history, user_msg)
            data = _parse_json_llm(raw)
            _log.debug("NavSelector LLM: %r → %s", user_msg, data)
        except Exception as exc:
            _log.warning("NavSelector LLM failed (%s); using keyword fallback", exc)
            data = _nav_select_fallback(user_msg, candidates, obj)
    else:
        # When llm_enable=False, auto-select first candidate (first choice selection)
        data = {"action": "navigate", "index": 0}

    action_type = data.get("action", "clarify")

    if action_type == "navigate":
        idx = int(data.get("index", 0))
        idx = max(0, min(idx, len(candidates) - 1))
        resp, nav_action = _navigate_to(candidates[idx])
        confirm = data.get("message") or resp
        return {
            "messages": updated_messages + [{"role": "assistant", "content": confirm}],
            "phase": PHASE_IDLE,
            "candidates": [],
            "response": confirm,
            "action": nav_action,
        }

    if action_type == "restart":
        msg = "Let's start over. What are you looking for?"
        return {
            "messages": updated_messages + [{"role": "assistant", "content": msg}],
            "phase": PHASE_IDLE,
            "goal": {},
            "candidates": [],
            "alternatives": [],
            "response": msg,
            "action": None,
        }

    # clarify — stay in NAV_SELECT
    clarify_msg = data.get("message") or _format_candidates(obj, candidates)
    return {
        "messages": updated_messages + [{"role": "assistant", "content": clarify_msg}],
        "phase": PHASE_NAV_SELECT,
        "candidates": candidates,
        "response": clarify_msg,
        "action": None,
    }


# ---------------------------------------------------------------------------
# Routing functions (conditional-edge selectors)
# ---------------------------------------------------------------------------


def _route_from_router(state: NavAgentState) -> str:
    phase = state.get("phase", PHASE_IDLE)
    if phase == PHASE_SUGGEST:
        return "suggest_classify"
    if phase == PHASE_NAV_SELECT:
        return "nav_selector"
    return "intent"


def _route_from_intent(state: NavAgentState) -> str:
    """Route based on intent_node output:
    - response set → fallback already emitted, done
    - phase=SUGGEST → implicit intent, emit suggestion
    - else → explicit intent, run search
    """
    if state.get("response"):
        return "done"
    if state.get("phase") == PHASE_SUGGEST:
        return "suggest_emit"
    return "search"


def _route_from_suggest_classify(state: NavAgentState) -> str:
    """Route based on suggest_classify_node output:
    - response set → FALLBACK or deny already emitted, done
    - goal has object → resolved to a label, run search
    - else → new_intent detected, re-enter intent_node
    """
    if state.get("response"):
        return "done"
    if (state.get("goal") or {}).get("object"):
        return "search"
    return "intent"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def _build_navigation_graph():
    """Compile the LangGraph navigation state machine."""
    b = StateGraph(NavAgentState)

    b.add_node("router", router_node)
    b.add_node("intent", intent_node)
    b.add_node("suggest_emit", suggest_emit_node)
    b.add_node("suggest_classify", suggest_classify_node)
    b.add_node("search", search_node)
    b.add_node("nav_selector", nav_selector_node)

    b.set_entry_point("router")

    b.add_conditional_edges(
        "router",
        _route_from_router,
        {
            "intent": "intent",
            "suggest_classify": "suggest_classify",
            "nav_selector": "nav_selector",
        },
    )

    b.add_conditional_edges(
        "intent",
        _route_from_intent,
        {
            "done": END,
            "suggest_emit": "suggest_emit",
            "search": "search",
        },
    )

    b.add_conditional_edges(
        "suggest_classify",
        _route_from_suggest_classify,
        {
            "done": END,
            "intent": "intent",
            "search": "search",
        },
    )

    b.add_edge("suggest_emit", END)
    b.add_edge("search", END)
    b.add_edge("nav_selector", END)

    return b.compile()


# ---------------------------------------------------------------------------
# Public agent class
# ---------------------------------------------------------------------------

_compiled_graph = None  # module-level singleton


def _get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = _build_navigation_graph()
    return _compiled_graph


class NavigationAgent:
    """Multi-turn conversational navigation assistant backed by a LangGraph graph.

    LLM nodes handle all conversation reasoning:
      - IntentNode (LLM)      — intent classification and entity extraction
      - SuggestNode (LLM)     — alternative suggestion and reply classification
      - NavSelectorNode (LLM) — multi-room candidate disambiguation

    VLM is used only inside the scene-graph query pipeline for slow visual
    re-ranking of candidate object images (slow_reasoning flag, unchanged).

    Usage::

        agent = NavigationAgent(
            scene_graph=graph,
            room_types=params.main.room_types,
            object_labels=label_list,
        )
        response, action = agent.process_message(user_text, llm_enable=True)
        if action:
            publish_navigation_goal(action)
    """

    def __init__(
        self,
        scene_graph: Any = None,
        room_types: Optional[List[str]] = None,  # kept for API compatibility
        object_labels: Optional[List[str]] = None,
        floor_labels: Optional[List[str]] = None,
    ) -> None:
        self._graph = _get_graph()
        self.scene_graph = scene_graph
        self.room_labels = room_types or []
        self.object_labels = object_labels or []
        self.floor_labels = floor_labels or self._infer_floor_labels(scene_graph)
        self._dialogue_state = self._empty_dialogue_state()

    @staticmethod
    def _infer_floor_labels(scene_graph: Any) -> List[str]:
        """Extract floor names from scene graph when floor_labels is not provided."""
        if scene_graph is None or not hasattr(scene_graph, "floors"):
            return []
        return [
            fl.name or f"Floor {i}" for i, fl in enumerate(scene_graph.floors) if fl is not None
        ]

    # ---- public ----

    def process_message(
        self,
        user_message: str,
        history=None,  # unused; kept for API compatibility with MotionAgent
        nav_state=None,  # unused; state is managed internally
        llm_enable: bool = True,
        slow_reasoning: bool = False,
    ) -> tuple:
        """Process one user turn → (response_text, action_dict | None)."""
        state: NavAgentState = {
            "user_message": user_message,
            "llm_enable": llm_enable,
            "slow_reasoning": slow_reasoning,
            "room_labels": self.room_labels,
            "object_labels": self.object_labels,
            "floor_labels": self.floor_labels,
            "scene_graph": self.scene_graph,
            "response": "",
            "action": None,
            **self._dialogue_state,
        }
        result = self._graph.invoke(state)
        self._dialogue_state = {
            "phase": result.get("phase", PHASE_IDLE),
            "goal": result.get("goal") or {},
            "alternatives": result.get("alternatives") or [],
            "candidates": result.get("candidates") or [],
            "messages": result.get("messages") or [],
        }
        return result.get("response") or FALLBACK, result.get("action")

    def extract_intent(
        self,
        user_message: str,
        history: Optional[List[Dict]] = None,
        llm_enable: bool = True,
    ) -> Dict:
        """Extract navigation intent without querying the scene graph.

        Useful for debugging LLM extraction in isolation.  Returns the raw
        intent dict from ``_extract_intent``:

          Explicit: {"intent": "explicit", "object": ..., "room": ..., "floor": ...}
          Implicit: {"intent": "implicit", "alternatives": [...]}
        """
        return _extract_intent(
            user_message,
            self.object_labels,
            self.room_labels,
            self.floor_labels,
            list(history or []),
            llm_enable=llm_enable,
        )

    def reset(self) -> None:
        """Reset dialogue state to IDLE (start a new conversation)."""
        self._dialogue_state = self._empty_dialogue_state()

    # ---- private ----

    @staticmethod
    def _empty_dialogue_state() -> dict:
        return {
            "phase": PHASE_IDLE,
            "goal": {},
            "alternatives": [],
            "candidates": [],
            "messages": [],
        }
