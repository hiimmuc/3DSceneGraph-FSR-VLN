"""LLM utilities for query parsing and inference with multi-provider support."""

import json
import os
import socket
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import AzureOpenAI, OpenAI

load_dotenv()

# Environment variable defaults and validation
_DEFAULT_OLLAMA_MODEL = "qwen3.5:latest"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def _validate_env_var(key: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default and validation.

    Args:
        key: Environment variable name.
        default: Default value if not set.

    Returns:
        Environment variable value or default.

    Raises:
        ValueError: If required variable is missing or is placeholder "xxxx".
    """
    value = os.environ.get(key, default)
    if value is None or value == "xxxx":
        raise ValueError(f"Missing or invalid env var: {key}")
    return value


def create_llm_client(provider: Optional[str] = None) -> Tuple[object, str]:
    """Create LLM client and return (client, model_name) tuple.

    Args:
        provider: Provider name ("azure" or "ollama"). If None, reads LLM_PROVIDER env var.

    Returns:
        Tuple of (client, model_name).

    Raises:
        ValueError: If provider is unknown or required env vars are missing.
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "azure":
        endpoint = _validate_env_var("AZURE_OPENAI_ENDPOINT")
        api_key = _validate_env_var("AZURE_OPENAI_API_KEY")
        api_version = _validate_env_var("AZURE_OPENAI_API_VERSION")
        model = _validate_env_var("AZURE_OPENAI_MODEL")

        return (
            AzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
            ),
            model,
        )
    elif provider == "ollama":
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        model = os.environ.get("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        return OpenAI(base_url=base_url, api_key="ollama"), model
    else:
        raise ValueError(f"Unknown LLM provider: {provider}. Choose 'azure' or 'ollama'.")


class Conversation:
    """Manages conversation state for LLM interactions."""

    def __init__(self, messages: List[dict], include_env_messages: bool = False) -> None:
        """Initialize conversation.

        Args:
            messages: List of message dictionaries.
            include_env_messages: Include environment-tagged messages in output.
        """
        self._messages = messages
        self._include_env_messages = include_env_messages

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation.

        Args:
            role: Message role (e.g., "user", "assistant", "system").
            content: Message content.
        """
        self._messages.append({"role": role, "content": content})

    @property
    def messages(self) -> List[dict]:
        """Get messages, excluding environment messages if configured."""
        if self._include_env_messages:
            return self._messages
        return [
            m for m in self._messages if m.get("role", "").lower() not in ["env", "environment"]
        ]

    @property
    def messages_including_env(self) -> List[dict]:
        """Get all messages including environment messages."""
        return self._messages


def send_query(
    client: object, messages: List[dict], model: str, temperature: float = 0.0, **kwargs
) -> object:
    """Send query to LLM.

    Args:
        client: LLM client (OpenAI or AzureOpenAI).
        messages: List of message dictionaries.
        model: Model name.
        temperature: Sampling temperature [0, 2].
        **kwargs: Additional arguments to pass to API.

    Returns:
        API response object.
    """
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


# Module-level cached LLM client (created once, reused for all queries)
_cached_client: Optional[object] = None
_cached_model: Optional[str] = None


def _get_cached_client() -> Tuple[object, str]:
    """Return a module-level singleton LLM client to avoid per-query reconnects."""
    global _cached_client, _cached_model
    if _cached_client is None:
        _cached_client, _cached_model = create_llm_client()
    return _cached_client, _cached_model


@lru_cache(maxsize=128)
def send_query_cached(model: str, messages_str: str, temperature: float = 0.0) -> str:
    """Send cached query to avoid redundant LLM calls for identical inputs.

    Args:
        model: Model name.
        messages_str: JSON-serialized messages list (used as cache key).
        temperature: Must be 0.0 for deterministic caching.

    Returns:
        Cached or freshly fetched response content string.
    """
    if temperature != 0.0:
        raise ValueError("Caching only works with temperature=0.0 for deterministic responses")
    import json as _json
    messages = _json.loads(messages_str)
    client, _ = _get_cached_client()
    response = send_query(client, messages, model, temperature=temperature)
    return response.choices[0].message.content.strip()


class QueryParser:
    """Parse hierarchical queries into floor, room, and object components."""

    QUERY_SPECS = {
        ("obj", "room", "floor"): "floor, room, and object",
        ("obj", "room"): "room and object",
        ("obj", "floor"): "floor and object",
        ("obj",): "object only",
    }

    def __init__(self, client: object = None, model: str = None):
        """Initialize parser with optional client/model (else uses cached singleton).

        Args:
            client: Optional pre-created LLM client.
            model: Optional model name override.
        """
        if client is None or model is None:
            client, model = _get_cached_client()
        self.client = client
        self.model = model

    def _build_system_prompt(self, spec_tuple: Tuple[str, ...]) -> str:
        """Build system prompt for given query specification."""
        spec_name = self.QUERY_SPECS.get(spec_tuple, "unknown")
        return f"You are a query parser. Parse the instruction into {spec_name}. If a component cannot be parsed, leave it empty."

    def _parse_response(
        self, response_str: str, spec_tuple: Tuple[str, ...]
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse LLM response into (floor, room, object).

        Args:
            response_str: Raw response from LLM.
            spec_tuple: Query spec tuple indicating what was parsed.

        Returns:
            Tuple of (floor, room, object), with None for unparsed components.
        """
        parts = [x.strip() for x in response_str.strip().rstrip("]").lstrip("[").split(",")]
        floor, room, obj = None, None, None

        try:
            if spec_tuple == ("obj", "room", "floor"):
                floor, room, obj = (parts + [None] * 3)[:3]
            elif spec_tuple == ("obj", "room"):
                room, obj = (parts + [None] * 2)[:2]
            elif spec_tuple == ("obj", "floor"):
                floor, obj = (parts + [None] * 2)[:2]
            elif spec_tuple == ("obj",):
                obj = parts[0] if parts else None
        except (IndexError, ValueError) as e:
            print(f"Warning: Failed to parse LLM result '{response_str}': {e}")

        return floor, room, obj

    # Keywords that suggest a room or floor component is present in the query
    _SPATIAL_HINTS = frozenset([
        "room", "floor", "level", "upstairs", "downstairs", "kitchen", "bedroom",
        "bathroom", "office", "living", "dining", "hallway", "garage", "basement",
        "corridor", "lobby", "entrance", "storage",
    ])

    def _has_spatial_hints(self, instruction: str) -> bool:
        words = set(instruction.lower().split())
        return bool(words & self._SPATIAL_HINTS)

    def parse(
        self, instruction: str, spec: Tuple[str, ...] = ("obj", "room", "floor")
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse instruction into hierarchy components.

        Fast-path: if spec is obj-only or no spatial hints are found in the
        instruction, skip the LLM and return the full instruction as the
        object query (avoiding ~8s Ollama round-trip).

        Args:
            instruction: User instruction to parse.
            spec: Tuple of components to parse (e.g., ("obj", "room", "floor")).

        Returns:
            Tuple of (floor, room, object).
        """
        if spec not in self.QUERY_SPECS:
            raise ValueError(
                f"Unknown query spec: {spec}. Available: {list(self.QUERY_SPECS.keys())}"
            )

        if spec == ("obj",):
            return None, None, instruction.strip()

        # Fast-path: no room/floor hints → treat full instruction as object query
        if not self._has_spatial_hints(instruction):
            # Strip common prefixes like "Find me the", "Where is the", etc.
            import re as _re
            obj = _re.sub(
                r"^(?:find(?:\s+me)?|where(?:'s|\s+is)|locate|show(?:\s+me)?)[\s,]+(?:the\s+)?",
                "",
                instruction.strip(),
                flags=_re.IGNORECASE,
            ).strip() or instruction.strip()
            print(f"Fast-path parsed '{instruction}' -> obj='{obj}'")
            return None, None, obj

        system_prompt = self._build_system_prompt(spec)
        user_prompt = f"Please parse: {instruction}\nOutput format: comma-separated list in order."

        import json as _json
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_result = send_query_cached(
            self.model, _json.dumps(messages, ensure_ascii=False), temperature=0.0
        )
        print(f"Parsed '{instruction}' -> '{raw_result}'")

        return self._parse_response(raw_result, spec)


def infer_room_type_from_objects(
    object_list: List[str], candidate_room_types: List[str] = None
) -> str:
    """Infer room type from a list of objects.

    Args:
        object_list: List of detected object names.
        candidate_room_types: Optional list of candidate room types to filter by.

    Returns:
        Inferred room type string.
    """
    client, model = create_llm_client()

    objects_str = ", ".join(object_list)
    constraint = ""
    if candidate_room_types:
        constraint = f"Please choose from: {', '.join(candidate_room_types)}."

    system_msg = "You are a room type classifier. Infer room type from detected objects. Answer only the room name."

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": "Objects: bed, wardrobe, chair. Room type?"},
        {"role": "assistant", "content": "Bedroom"},
        {"role": "user", "content": f"Objects: {objects_str}. {constraint} Room type?"},
    ]

    response = send_query(client, messages, model, temperature=0.0)
    room_type = response.choices[0].message.content.strip()
    print(f"Inferred room type: {room_type}")
    return room_type


# Backward-compatible wrapper functions
def infer_floor_id_from_query(floor_ids_list: List[int], query: str) -> int:
    """Infer floor ID from a natural-language query using LLM.

    Args:
        floor_ids_list: List of valid floor IDs (e.g., [1, 2, 3]).
        query: User query string containing floor reference.

    Returns:
        Matched floor ID from floor_ids_list, or first ID if inference fails.
    """
    client, model = create_llm_client()

    floors_str = ", ".join(str(f) for f in floor_ids_list)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a floor identifier. Given a query, return only the floor number "
                "(integer) from the available floors. Answer with just the number."
            ),
        },
        {
            "role": "user",
            "content": f"Available floors: {floors_str}. Query: {query}. Which floor?",
        },
    ]

    response = send_query(client, messages, model, temperature=0.0)
    result = response.choices[0].message.content.strip()

    try:
        floor_id = int(result)
        if floor_id in floor_ids_list:
            return floor_id
    except ValueError:
        pass

    # Default to first floor if parsing fails
    return floor_ids_list[0] if floor_ids_list else 1


def parse_hierarchy_query(
    cfg, instruction: str, parser: "QueryParser" = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a natural language instruction into (floor, room, object) components."""
    if parser is None:
        parser = QueryParser()
    spec = tuple(cfg.main.long_query.spec) if hasattr(cfg, "main") else ("obj", "room", "floor")
    return parser.parse(instruction, spec)


# Backward-compatible alias
parse_hier_query_use_prompt_insentence_parse_icra = parse_hierarchy_query


def parse_floor_room_object_gpt40(instruction: str) -> Tuple[str, str, str]:
    """Legacy wrapper for floor/room/object parsing."""
    parser = QueryParser()
    floor, room, obj = parser.parse(instruction, ("obj", "room", "floor"))
    return floor or "", room or "", obj or ""


# ---------------------------------------------------------------------------
# MotionAgent — conversational robot assistant
# ---------------------------------------------------------------------------

# Implicit object mapping: lifestyle keywords → object categories
_IMPLICIT_OBJECT_MAP = {
    "sleepy": "bed",
    "tired": "bed",
    "sleep": "bed",
    "rest": "bed",
    "nap": "bed",
    "bored": "entertainment",
    "boring": "entertainment",
    "entertain": "TV",
    "entertainment": "TV",
    "watch": "TV",
    "movie": "TV",
    "movies": "TV",
    "film": "TV",
    "show": "TV",
    "music": "radio",
    "hungry": "kitchen",
    "eat": "kitchen",
    "food": "kitchen",
    "thirsty": "kitchen",
    "drink": "kitchen",
    "work": "desk",
    "study": "desk",
    "read": "bookshelf",
    "book": "bookshelf",
    "sit": "chair",
    "relax": "sofa",
}

# Navigation dialogue states
_NAV_STATE_INFER = "INFER"
_NAV_STATE_CLARIFY = "CLARIFY"
_NAV_STATE_CONFIRM = "CONFIRM"
_NAV_STATE_DONE = "DONE"


def publish_navigation_goal(
    goal: Dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 5005,
) -> bool:
    """Send a navigation goal to the ROS2 goal bridge via UDP.

    Args:
        goal: Dict with keys ``name``, ``x``, ``y``, ``z``.
        host: UDP host (default ``127.0.0.1``).
        port: UDP port (default ``5005`` — matches ros2_goal_bridge.py).

    Returns:
        True on success, False on failure.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(json.dumps(goal).encode("utf-8"), (host, port))
        sock.close()
        return True
    except Exception as e:
        import logging
        logging.getLogger(__name__).error("publish_navigation_goal failed: %s", e)
        return False


class MotionAgent:
    """Conversational robot assistant named 'Motion'.

    Classifies user intent as general conversation or navigation request and
    handles multi-turn clarification dialogues to resolve the navigation target
    before publishing a ROS2 goal position.

    Usage::

        agent = MotionAgent(scene_graph=graph_instance)
        response, action = agent.process_message(user_text, history, nav_state)
        if action:
            publish_navigation_goal(action)

    The ``nav_state`` dict is mutated in-place across turns to track the
    navigation dialogue state machine (``INFER → CLARIFY → CONFIRM → DONE``).
    Pass ``{}`` initially and persist it in Streamlit ``session_state``.
    """

    _SYSTEM_PROMPT = (
        "You are Motion, a friendly and helpful robot assistant. "
        "You live in a smart building and can guide people to any object or location. "
        "Keep responses concise and warm. "
        "When someone mentions a need or feeling, infer what they might need "
        "(e.g. 'sleepy' → bed, 'bored' → TV or radio, 'hungry' → kitchen). "
        "Always speak as Motion."
    )

    _INTENT_PROMPT = (
        "Classify the user's last message as either 'general' or 'navigation'.\n"
        "'navigation' means the user wants to go somewhere, find an object, "
        "or you infer they need something physical (e.g. bed, TV, kitchen).\n"
        "'general' means casual conversation with no navigation need.\n"
        "Reply with exactly one word: general  or  navigation."
    )

    def __init__(
        self,
        scene_graph: Optional[Any] = None,
        client: Optional[Any] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize MotionAgent.

        Args:
            scene_graph: Loaded ``Graph`` instance for scene queries (optional).
            client: Pre-created LLM client (optional — uses cached singleton).
            model: Model name override (optional).
        """
        if client is None or model is None:
            self._client, self._model = _get_cached_client()
        else:
            self._client = client
            self._model = model
        self.scene_graph = scene_graph

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_message(
        self,
        user_message: str,
        history: List[Dict[str, str]],
        nav_state: Dict[str, Any],
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Process one user turn and return (response_text, action_or_None).

        Args:
            user_message: The user's latest message.
            history: Conversation history as list of ``{role, content}`` dicts
                (roles: ``"user"`` or ``"assistant"``).  Mutated in-place.
            nav_state: Navigation state dict (mutated in-place).  Pass ``{}``
                for the first message.

        Returns:
            Tuple of:
            - ``response_text``: Motion's reply to display/speak.
            - ``action``: ``{name, x, y, z}`` dict when a navigation target is
              resolved and should be published, else ``None``.
        """
        # If we're in the middle of a navigation dialogue, continue it
        if nav_state.get("state") in (_NAV_STATE_CLARIFY, _NAV_STATE_CONFIRM):
            return self._continue_navigation(user_message, history, nav_state)

        intent = self._classify_intent(user_message, history)

        if intent == "navigation":
            return self._start_navigation(user_message, history, nav_state)
        else:
            response = self._general_chat(user_message, history)
            return response, None

    # ------------------------------------------------------------------
    # Intent classification
    # ------------------------------------------------------------------

    def _classify_intent(self, message: str, history: List[Dict]) -> str:
        """Return 'navigation' or 'general' for the given message."""
        # Fast-path: keyword check for implicit object mapping
        words = set(message.lower().split())
        if words & set(_IMPLICIT_OBJECT_MAP.keys()):
            return "navigation"

        # Ask LLM
        messages = [{"role": "system", "content": self._INTENT_PROMPT}]
        # Include last 4 turns for context
        for turn in history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        try:
            response = send_query(self._client, messages, self._model, temperature=0.0)
            result = response.choices[0].message.content.strip().lower()
            return "navigation" if "navigation" in result else "general"
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("Intent classification failed: %s", e)
            return "general"

    # ------------------------------------------------------------------
    # General conversation
    # ------------------------------------------------------------------

    def _general_chat(self, message: str, history: List[Dict]) -> str:
        """Generate a general conversational response."""
        messages = [{"role": "system", "content": self._SYSTEM_PROMPT}]
        for turn in history[-10:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        try:
            response = send_query(self._client, messages, self._model, temperature=0.7)
            return response.choices[0].message.content.strip()
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("General chat failed: %s", e)
            return "I'm sorry, I had trouble responding. Could you say that again?"

    # ------------------------------------------------------------------
    # Navigation dialogue
    # ------------------------------------------------------------------

    def _infer_object_query(self, message: str, history: List[Dict]) -> str:
        """Map the user message to an object search query."""
        words = message.lower().split()
        for word in words:
            if word in _IMPLICIT_OBJECT_MAP:
                return _IMPLICIT_OBJECT_MAP[word]

        # Ask LLM to extract the target object
        prompt = (
            "The user wants to navigate to something. "
            "Extract the most likely physical object or location they need. "
            "Reply with just the object name (e.g. 'bed', 'TV', 'kitchen'). "
            f"User said: {message}"
        )
        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            r = send_query(self._client, messages, self._model, temperature=0.0)
            return r.choices[0].message.content.strip().lower()
        except Exception:
            return message.strip()

    def _query_scene_graph(self, object_query: str, top_k: int = 5) -> List[Dict]:
        """Query the scene graph and return a list of candidate result dicts."""
        if self.scene_graph is None:
            return []
        try:
            _floor, rooms, objects, res_dict = self.scene_graph.query_hierarchy(
                object_query, top_k=top_k
            )
            candidates = []
            _T_SWITCH = __import__("numpy").array(
                [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]],
                dtype=float,
            )
            _T_TO_MAP = __import__("numpy").linalg.inv(_T_SWITCH)
            import numpy as np

            for obj, room in zip(objects, rooms):
                center_sg = np.array(obj.pcd.get_center())
                center_map = (_T_TO_MAP @ np.hstack((center_sg, 1.0)))[:3]
                floor_label = "?"
                obj_id_parts = str(obj.object_id).split("_")
                if obj_id_parts and self.scene_graph.floors:
                    idx = int(obj_id_parts[0]) if obj_id_parts[0].isdigit() else -1
                    if 0 <= idx < len(self.scene_graph.floors):
                        fl = self.scene_graph.floors[idx]
                        floor_label = fl.name or f"Floor {idx}"
                room_label = room.name or room.room_id
                candidates.append(
                    {
                        "name": obj.name,
                        "room": room_label,
                        "floor": floor_label,
                        "x": float(center_map[0]),
                        "y": float(center_map[1]),
                        "z": float(center_map[2]),
                        "object_id": obj.object_id,
                    }
                )
            return candidates
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Scene graph query failed: %s", e)
            return []

    def _start_navigation(
        self,
        message: str,
        history: List[Dict],
        nav_state: Dict,
    ) -> Tuple[str, Optional[Dict]]:
        """Begin a navigation dialogue from a fresh user message."""
        object_query = self._infer_object_query(message, history)
        nav_state["object_query"] = object_query
        candidates = self._query_scene_graph(object_query)
        nav_state["candidates"] = candidates

        if not candidates:
            nav_state.clear()
            response = self._general_chat(
                message
                + "\n(Note: I couldn't find any matching objects in my scene map. "
                "Please let the user know gently.)",
                history,
            )
            return response, None

        # Single candidate — ask for confirmation
        if len(candidates) == 1:
            c = candidates[0]
            nav_state["state"] = _NAV_STATE_CONFIRM
            nav_state["selected"] = c
            response = (
                f"I found a {c['name']} in {c['room']}, {c['floor']}. "
                "Shall I take you there?"
            )
            return response, None

        # Check if all candidates are in the same room — skip clarification
        rooms_seen = {(c["room"], c["floor"]) for c in candidates}
        if len(rooms_seen) == 1:
            c = candidates[0]
            nav_state["state"] = _NAV_STATE_CONFIRM
            nav_state["selected"] = c
            response = (
                f"I found a {c['name']} in {c['room']}, {c['floor']}. "
                "Shall I take you there?"
            )
            return response, None

        # Multiple locations — ask user to pick
        nav_state["state"] = _NAV_STATE_CLARIFY
        options_text = "\n".join(
            f"  {i + 1}. {c['name']} — {c['room']}, {c['floor']}"
            for i, c in enumerate(candidates)
        )
        response = (
            f"I found {len(candidates)} options:\n{options_text}\n"
            "Which one would you like to go to? (say the number or describe the location)"
        )
        return response, None

    def _continue_navigation(
        self,
        message: str,
        history: List[Dict],
        nav_state: Dict,
    ) -> Tuple[str, Optional[Dict]]:
        """Continue an in-progress navigation clarification dialogue."""
        state = nav_state.get("state")

        if state == _NAV_STATE_CONFIRM:
            selected = nav_state.get("selected", {})
            if self._user_confirmed(message):
                nav_state["state"] = _NAV_STATE_DONE
                nav_state.clear()
                action = {
                    "name": selected.get("name", "target"),
                    "x": selected["x"],
                    "y": selected["y"],
                    "z": selected["z"],
                }
                response = (
                    f"Great! Let me take you to the {selected.get('name', 'target')} "
                    f"in {selected.get('room', '')}. I'm on my way!"
                )
                return response, action
            else:
                nav_state.clear()
                return "Understood, I won't navigate there. Let me know if you need anything else!", None

        if state == _NAV_STATE_CLARIFY:
            candidates = nav_state.get("candidates", [])
            selected = self._resolve_selection(message, candidates)
            if selected is not None:
                nav_state["state"] = _NAV_STATE_CONFIRM
                nav_state["selected"] = selected
                response = (
                    f"Got it! Taking you to the {selected['name']} "
                    f"in {selected['room']}, {selected['floor']}. Shall I confirm?"
                )
                return response, None
            else:
                options_text = "\n".join(
                    f"  {i + 1}. {c['name']} — {c['room']}, {c['floor']}"
                    for i, c in enumerate(candidates)
                )
                response = (
                    "Sorry, I didn't catch that. Please choose a number or describe the location:\n"
                    + options_text
                )
                return response, None

        # Fallback — clear stale state
        nav_state.clear()
        return self._general_chat(message, history), None

    def _user_confirmed(self, message: str) -> bool:
        """Return True if the message is an affirmative response."""
        affirmatives = {"yes", "yeah", "yep", "sure", "ok", "okay", "please", "go", "do it", "confirm", "y"}
        msg_lower = message.strip().lower()
        return any(a in msg_lower for a in affirmatives)

    def _resolve_selection(
        self, message: str, candidates: List[Dict]
    ) -> Optional[Dict]:
        """Try to match user's selection to one of the candidates."""
        msg_lower = message.strip().lower()

        # Numeric selection
        for i, c in enumerate(candidates):
            if str(i + 1) in msg_lower:
                return c

        # Location-based match
        for c in candidates:
            if c["room"].lower() in msg_lower or c["floor"].lower() in msg_lower:
                return c

        # LLM-assisted disambiguation
        try:
            opts = "; ".join(
                f"{i + 1}: {c['name']} in {c['room']}, {c['floor']}"
                for i, c in enumerate(candidates)
            )
            prompt = (
                f"The user said: '{message}'. "
                f"Available options: {opts}. "
                "Which option number did they choose? Reply with just the number, "
                "or 'none' if unclear."
            )
            r = send_query(
                self._client,
                [{"role": "user", "content": prompt}],
                self._model,
                temperature=0.0,
            )
            result = r.choices[0].message.content.strip()
            if result.isdigit():
                idx = int(result) - 1
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass

        return None


if __name__ == "__main__":
    # Test example
    parser = QueryParser()
    result = parser.parse("sofa in living room on floor 1")
    print(f"Result: {result}")
