# FSR-VLN Navigation Tools

LangGraph-based dialogue agent for indoor robot navigation.  
LLM plays two distinct roles in the pipeline: **Retrieval** (understanding what the user wants) and **Selector** (choosing among multiple matching locations).

---

## Architecture Overview

```text
START
  │
  ▼
router ──────────────────────────────────────────────────
  │ IDLE              │ SUGGEST          │ NAV_SELECT
  ▼                   ▼                  ▼
intent_node    suggest_classify     nav_selector ──► END
  │                   │
  │ explicit           │ choice/ambiguous ──► search_node ──► END
  │                   │
  │ implicit           │ new_intent ──► intent_node (re-enter)
  │
  ▼
suggest_emit ──► END   (phase=SUGGEST, returns clarification question)

search_node outcomes:
  same_room:  navigate directly   (action set, phase=IDLE)
  not_found:  ask to retry        (phase=IDLE)
  multi_room: present list        (phase=NAV_SELECT)
```

---

## LLM as Retrieval

The retrieval role covers **understanding user intent** and **grounding it to available scene labels** before any scene-graph query is made.

### IntentNode — `intent_node` in `agents.py`

**System prompt**: `make_intent_node_system()` in `prompts.py`  
**LLM call**: `_llm_call(system, history, user_message)` → `send_query_cached()`

The LLM receives:

- The current user utterance + episodic conversation history.
- Available `object_labels`, `room_labels`, and `floor_labels` from the scene graph.

It classifies the utterance and extracts a navigation goal as structured JSON:

| Intent type | When | LLM output |
|-------------|------|-----------|
| `explicit`  | User directly names an object or room | `{"intent":"explicit","object":<label>,"room":<label\|null>,"floor":<label\|null>}` |
| `implicit`  | User expresses state/feeling/activity | `{"intent":"implicit","alternatives":[<label>, ...]}` |

**Grounding constraint**: all output labels must come from the provided lists — no hallucination.  
**Safety guard**: if the LLM says `explicit` but the extracted label does not literally appear in the user message, the intent is downgraded to `implicit` to force a confirmation step.

#### Implicit intent examples

```
"I am sleepy"     → {"intent": "implicit", "alternatives": ["bed", "sofa", "pillow"]}
"I need to work"  → {"intent": "implicit", "alternatives": ["desk", "chair"]}
"I'm hungry"      → {"intent": "implicit", "alternatives": ["refrigerator", "table"]}
```

### QueryParser — `QueryParser` in `llm.py`

**System prompt**: `QUERY_PARSER_SYSTEM` in `prompts.py`

Used when no `object_labels` inventory is provided (unconstrained mode). The LLM parses the instruction into `(floor, room, object)` components and optionally returns `objects_alt` for implicit queries.  
Results are cached via `send_query_cached()` (LRU, `temperature=0.0`).

### Synonym Resolvers — `ROOM_SYNONYM_SYSTEM` / `OBJ_SYNONYM_SYSTEM` in `prompts.py`

After extraction, synonyms in the user's utterance are normalised to exact scene-graph labels:

```text
"lounge"   → "living room"   (room synonym)
"couch"    → "sofa"          (object synonym)
"TV"       → "TV"            (exact match, passthrough)
```

---

## LLM as Selector

The selector role covers **choosing among multiple matches** once the scene graph has been queried. Two nodes participate.

### SuggestClassifyNode — `suggest_classify_node` in `agents.py`

**System prompt**: `make_suggest_classify_system()` in `prompts.py`  
**Triggered when**: `phase == SUGGEST` (IntentNode returned implicit intent, robot asked a clarification question).

The LLM classifies the user's reply to the clarification question:

| Output type | Meaning | Next step |
|-------------|---------|-----------|
| `explicit_choice` | User clearly chose one alternative | Set goal → `search_node` |
| `ambiguous` | User agreed, said "any", or unclear | Use `alternatives[0]` → `search_node` |
| `new_intent` | User expressed a different goal | Clear state → re-enter `intent_node` |

```json
// User reply: "the bed please"
{"type": "explicit_choice", "object": "bed"}

// User reply: "whatever is closest"
{"type": "ambiguous"}

// User reply: "never mind, find me a desk"
{"type": "new_intent"}
```

**Constraint**: the chosen label must come from the `alternatives` list — no new labels invented.

### NavSelectorNode — `nav_selector_node` in `agents.py`

**System prompt**: `make_nav_selector_system()` in `prompts.py`  
**Triggered when**: `phase == NAV_SELECT` (scene-graph returned multiple candidate locations for the same object type).

The LLM receives:

- Full candidate list (re-injected every turn with `[index] name — room, floor` format).
- Full episodic conversation history (for reference resolution: "the other one", "the one on floor 3").

It decides:

| Action | Condition | Output |
|--------|-----------|--------|
| `navigate` | User clearly selected, said "any", or doesn't care | `{"action":"navigate","index":<int>,"message":<confirmation>}` |
| `clarify` | Multiple candidates still match | `{"action":"clarify","message":<one question>}` |
| `restart` | User wants to start over | `{"action":"restart"}` |

```json
// User: "the one on floor 2"
{"action": "navigate", "index": 1, "message": "I will navigate you to the desk in office, Floor 2."}

// User: "which is nearer?"
{"action": "clarify", "message": "The first one is in room 101 on floor 1, the second in room 301 on floor 3. Which do you prefer?"}
```

---

## Data Flow Summary

```
User utterance
      │
      ▼
[LLM — RETRIEVAL]
IntentNode / QueryParser
  · Classify explicit vs implicit
  · Extract {object, room?, floor?}
  · Ground to scene-graph label lists
      │
      ├─ implicit ──► SuggestEmitNode (template/LLM clarification question)
      │                    │
      │              User reply
      │                    │
      │              [LLM — SELECTOR]
      │              SuggestClassifyNode
      │                · explicit_choice / ambiguous / new_intent
      │
      ▼
SearchNode  (scene-graph query, VLM re-ranking if slow_reasoning=True)
      │
      ├─ 1 result  ──► navigate directly
      ├─ 0 results ──► ask to retry
      │
      └─ N results ──► [LLM — SELECTOR]
                       NavSelectorNode (multi-turn)
                         · navigate / clarify / restart
```

---

## Module Reference

| File | Purpose |
|------|---------|
| `agents.py` | LangGraph state machine — all nodes and routing logic |
| `llm.py` | OpenAI-compatible client, `QueryParser`, `send_query_cached`, connection probe |
| `prompts.py` | All system-prompt strings and builder functions |
| `vlm.py` | OWLv2 vision-language model — local grounded object detection for VLM re-ranking |

---

## Configuration

| Environment variable | Default | Description |
|----------------------|---------|-------------|
| `HTTP_ENDPOINT` | `http://localhost:8000/v1` | OpenAI-compatible LLM endpoint |
| `MODEL_NAME` | `Qwen/Qwen3-VL-4B-Instruct` | Model to use for all LLM calls |
| `API_KEY` | `EMPTY` | API key (use `EMPTY` for local vLLM servers) |
| `LLM_PROVIDER` | `openai-compatible` | Provider label (logging only) |

### Runtime flags (per-turn in `NavAgentState`)

| Flag | Effect |
|------|--------|
| `llm_enable=True` | Full LLM retrieval + selector pipeline active |
| `llm_enable=False` | Keyword matching + template fallbacks only |
| `slow_reasoning=True` | Enables VLM re-ranking in `search_node` and LLM-phrased suggestions in `suggest_emit_node` |
| `slow_reasoning=False` (default) | Deterministic template suggestions; faster |
