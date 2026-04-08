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
    """Parse hierarchical queries into floor, room, and object components.

    Uses a structured JSON-output prompt so the LLM response is machine-readable
    without fragile comma-splitting, and applies synonym normalisation before
    returning results.
    """

    QUERY_SPECS = {
        ("obj", "room", "floor"): "floor, room, and object",
        ("obj", "room"): "room and object",
        ("obj", "floor"): "floor and object",
        ("obj",): "object only",
    }

    # Canonical room-type synonyms: any alias → canonical form sent to the graph
    _ROOM_SYNONYMS: Dict[str, str] = {
        "living room": "living room",
        "lounge": "living room",
        "family room": "living room",
        "sitting room": "living room",
        "bed room": "bedroom",
        "master bedroom": "bedroom",
        "guest room": "bedroom",
        "bath room": "bathroom",
        "restroom": "bathroom",
        "toilet": "bathroom",
        "lavatory": "bathroom",
        "wc": "bathroom",
        "study": "office",
        "work room": "office",
        "home office": "office",
        "dining room": "dining room",
        "dining area": "dining room",
        "eat-in kitchen": "kitchen",
        "kitchenette": "kitchen",
        "pantry": "kitchen",
        "hallway": "hallway",
        "hall": "hallway",
        "corridor": "hallway",
        "passage": "hallway",
        "entrance": "entrance",
        "entryway": "entrance",
        "foyer": "entrance",
        "lobby": "entrance",
        "storage room": "storage",
        "storeroom": "storage",
        "closet": "storage",
        "utility room": "storage",
        "laundry room": "laundry",
        "laundry": "laundry",
        "garage": "garage",
        "car port": "garage",
        "basement": "basement",
        "cellar": "basement",
        "attic": "attic",
        "loft": "attic",
        "balcony": "balcony",
        "terrace": "balcony",
        "patio": "balcony",
    }

    # Keyword → canonical object label
    _OBJ_SYNONYMS: Dict[str, str] = {
        "couch": "sofa",
        "settee": "sofa",
        "loveseat": "sofa",
        "telly": "TV",
        "television": "TV",
        "monitor": "computer monitor",
        "screen": "computer monitor",
        "fridge": "refrigerator",
        "cooler": "refrigerator",
        "icebox": "refrigerator",
        "bin": "trash can",
        "rubbish bin": "trash can",
        "garbage can": "trash can",
        "waste bin": "trash can",
        "wardrobe": "wardrobe",
        "closet": "wardrobe",
        "dresser": "chest of drawers",
        "chest of drawers": "chest of drawers",
        "nightstand": "bedside table",
        "night table": "bedside table",
        "bedside cabinet": "bedside table",
        "armchair": "chair",
        "recliner": "chair",
        "stool": "chair",
        "laptop": "laptop computer",
        "notebook computer": "laptop computer",
        "tap": "sink",
        "basin": "sink",
        "washbasin": "sink",
        "loo": "toilet",
        "commode": "toilet",
        "bookcase": "bookshelf",
        "book rack": "bookshelf",
        "desk lamp": "lamp",
        "floor lamp": "lamp",
        "table lamp": "lamp",
        "light": "lamp",
    }

    # Spatial keywords that hint at a room or floor component
    _SPATIAL_HINTS = frozenset(
        [
            "room", "floor", "level", "upstairs", "downstairs",
            "kitchen", "bedroom", "bathroom", "office", "living",
            "dining", "hallway", "garage", "basement", "corridor",
            "lobby", "entrance", "storage", "laundry", "attic",
            "balcony", "lounge", "study", "foyer", "cellar", "loft",
        ]
    )

    # Regex that strips common navigation prefixes before the object name
    import re as _re_module
    _PREFIX_RE = _re_module.compile(
        r"^(?:"
        r"find(?:\s+me)?|where(?:'s|\s+is)|locate|show(?:\s+me)?"
        r"|bring(?:\s+me)?|get(?:\s+me)?|take(?:\s+me\s+to)?"
        r"|navigate(?:\s+to)?|go(?:\s+to)?|i\s+need"
        r")[\s,]+(?:the\s+|a\s+|an\s+)?",
        _re_module.IGNORECASE,
    )

    _SYSTEM_PROMPT = (
        "You are a precise indoor-navigation query parser for a robotic assistant.\n"
        "Given a natural language instruction, extract the most likely navigation target.\n"
        "\n"
        "Rules:\n"
        "  - \"object\": the physical item, furniture, or room the user wants to reach.\n"
        "      * Use a generic category name (e.g. 'sofa', 'TV', 'bed', 'kitchen').\n"
        "      * For IMPLICIT intentions, infer the best single object:\n"
        "          'sleepy/tired/exhausted' → bed\n"
        "          'bored'                 → TV\n"
        "          'hungry/starving'       → kitchen\n"
        "          'thirsty'               → kitchen\n"
        "          'need to work/study'    → desk\n"
        "          'want to read'          → bookshelf\n"
        "          'want to relax'         → sofa\n"
        "          'need to wash/hygiene'  → sink\n"
        "  - \"objects_alt\": optional JSON array of 1-3 alternative object categories\n"
        "      ranked by likelihood (for implicit/ambiguous intents). Omit for explicit queries.\n"
        "  - \"room\": the room type if mentioned or strongly implied, else null.\n"
        "  - \"floor\": the floor/level if mentioned or strongly implied, else null.\n"
        "\n"
        "Return ONLY a JSON object. Do not add explanation, markdown, or extra keys.\n"
        "Required keys: \"object\", \"room\", \"floor\".\n"
        "Optional key:  \"objects_alt\" (array of strings).\n"
        "\n"
        "Examples:\n"
        "  \"Find the sofa in the living room on floor 2\" →\n"
        "    {\"object\":\"sofa\",\"room\":\"living room\",\"floor\":\"2\"}\n"
        "  \"Where is the TV?\" →\n"
        "    {\"object\":\"TV\",\"room\":null,\"floor\":null}\n"
        "  \"I am so sleepy\" →\n"
        "    {\"object\":\"bed\",\"objects_alt\":[\"pillow\",\"sofa\"],\"room\":null,\"floor\":null}\n"
        "  \"I am so bored\" →\n"
        "    {\"object\":\"TV\",\"objects_alt\":[\"radio\",\"bookshelf\"],\"room\":null,\"floor\":null}\n"
        "  \"I'm hungry\" →\n"
        "    {\"object\":\"kitchen\",\"objects_alt\":[\"dining table\",\"refrigerator\"],\"room\":\"kitchen\",\"floor\":null}\n"
        "  \"laptop upstairs\" →\n"
        "    {\"object\":\"laptop computer\",\"room\":null,\"floor\":\"upstairs\"}\n"
    )

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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_room(self, room: Optional[str]) -> Optional[str]:
        """Map room aliases to canonical room-type strings."""
        if not room:
            return None
        key = room.strip().lower()
        return self._ROOM_SYNONYMS.get(key, room.strip().lower())

    def _normalize_object(self, obj: Optional[str]) -> Optional[str]:
        """Map object aliases to canonical object-category strings."""
        if not obj:
            return None
        key = obj.strip().lower()
        return self._OBJ_SYNONYMS.get(key, obj.strip())

    def _strip_prefix(self, instruction: str) -> str:
        """Remove common navigation-verb prefixes from an instruction."""
        stripped = self._PREFIX_RE.sub("", instruction.strip()).strip()
        return stripped or instruction.strip()

    def _has_spatial_hints(self, instruction: str) -> bool:
        words = set(instruction.lower().split())
        return bool(words & self._SPATIAL_HINTS)

    def _parse_json_response(
        self, raw: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Extract (floor, room, object, alt_objects) from a JSON string.

        Returns:
            (floor, room, primary_object, alt_objects_list)
            Falls back to (None, None, None, []) on any parse error.
        """
        import json as _json
        import re as _re

        # Strip markdown code fences if the model wrapped the JSON
        cleaned = _re.sub(r"```(?:json)?\s*|\s*```", "", raw.strip())
        # Extract the first JSON object in the string
        m = _re.search(r"\{.*\}", cleaned, _re.DOTALL)
        if not m:
            print(f"Warning: no JSON object found in LLM response: {raw!r}")
            return None, None, None, []
        try:
            data = _json.loads(m.group())
        except _json.JSONDecodeError as exc:
            print(f"Warning: JSON decode error ({exc}) in: {m.group()!r}")
            return None, None, None, []

        def _clean(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            return None if s.lower() == "null" or s == "" else s

        floor = _clean(data.get("floor"))
        room  = _clean(data.get("room"))
        obj   = _clean(data.get("object"))

        # Optional alternative objects for implicit intents
        alts_raw = data.get("objects_alt", [])
        alt_objects: List[str] = []
        if isinstance(alts_raw, list):
            for a in alts_raw:
                cleaned_a = _clean(a)
                if cleaned_a:
                    alt_objects.append(cleaned_a)

        return floor, room, obj, alt_objects

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(
        self, instruction: str, spec: Tuple[str, ...] = ("obj", "room", "floor")
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Parse instruction into (floor, room, object) hierarchy components.

        For callers that also need alternative object candidates (implicit intents),
        use :meth:`parse_with_alts` instead.

        Fast-path: if spec is obj-only or no spatial hints are detected,
        skip the LLM call and return the stripped instruction as the object
        query (avoids Ollama round-trip latency).

        Args:
            instruction: User instruction to parse.
            spec: Tuple of components to parse (e.g., ("obj", "room", "floor")).

        Returns:
            Tuple of (floor, room, object) with None for absent components.
        """
        floor, room, obj, _alts = self.parse_with_alts(instruction, spec)
        return floor, room, obj

    def parse_with_alts(
        self, instruction: str, spec: Tuple[str, ...] = ("obj", "room", "floor")
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Parse instruction and return (floor, room, primary_object, alt_objects).

        ``alt_objects`` is a ranked list of alternative object categories for
        implicit-intent queries (e.g. 'I am sleepy' → alts=['pillow', 'sofa']).
        It is empty for explicit object queries.
        """
        if spec not in self.QUERY_SPECS:
            raise ValueError(
                f"Unknown query spec: {spec}. Available: {list(self.QUERY_SPECS.keys())}"
            )

        if spec == ("obj",):
            return None, None, self._normalize_object(self._strip_prefix(instruction)), []

        # Implicit-intent fast-path: check _IMPLICIT_OBJECT_MAP before calling LLM
        words = instruction.lower().split()
        for word in words:
            if word in _IMPLICIT_OBJECT_MAP:
                candidates = [
                    self._normalize_object(o) for o in _IMPLICIT_OBJECT_MAP[word]
                ]
                primary = candidates[0] if candidates else self._strip_prefix(instruction)
                alts = candidates[1:] if len(candidates) > 1 else []
                print(f"Implicit fast-path '{instruction}' [{word}] -> obj='{primary}'  alts={alts}")
                return None, None, primary, alts

        # Explicit spatial fast-path: no room/floor hints → single object, no alts
        if not self._has_spatial_hints(instruction):
            obj = self._normalize_object(self._strip_prefix(instruction))
            print(f"Fast-path parsed '{instruction}' -> obj='{obj}'")
            return None, None, obj, []

        import json as _json

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
        ]
        raw_result = send_query_cached(
            self.model, _json.dumps(messages, ensure_ascii=False), temperature=0.0
        )
        print(f"LLM parsed '{instruction}' -> '{raw_result}'")

        floor, room, obj, alt_objects = self._parse_json_response(raw_result)

        # Normalise and fall back to stripped instruction if object is empty
        floor = floor.strip() if floor else None
        room = self._normalize_room(room)
        obj = self._normalize_object(obj) or self._strip_prefix(instruction)
        alt_objects = [
            self._normalize_object(a) for a in alt_objects if a
        ]

        print(f"  -> floor={floor!r}  room={room!r}  obj={obj!r}  alts={alt_objects}")
        return floor, room, obj, alt_objects


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

# Implicit object mapping: lifestyle keywords → ranked list of candidate object categories.
# The list is ordered from most to least likely so multi-query scene-graph searches
# return the best match first.
_IMPLICIT_OBJECT_MAP: Dict[str, List[str]] = {
    # Fatigue / sleep
    "sleepy":      ["bed", "pillow", "sofa", "mattress"],
    "tired":       ["bed", "sofa", "pillow", "chair"],
    "exhausted":   ["bed", "sofa", "pillow"],
    "sleep":       ["bed", "pillow", "mattress"],
    "rest":        ["bed", "sofa", "chair"],
    "nap":         ["bed", "sofa", "pillow"],
    # Boredom / entertainment
    "bored":       ["TV", "radio", "bookshelf", "sofa"],
    "boring":      ["TV", "radio", "bookshelf"],
    "entertain":   ["TV", "radio", "bookshelf"],
    "entertainment":["TV", "radio", "bookshelf"],
    "watch":       ["TV"],
    "movie":       ["TV"],
    "movies":      ["TV"],
    "film":        ["TV"],
    "show":        ["TV"],
    # Music
    "music":       ["radio", "TV", "speaker"],
    "listen":      ["radio", "speaker"],
    # Hunger / thirst
    "hungry":      ["kitchen", "dining table", "refrigerator"],
    "starving":    ["kitchen", "dining table", "refrigerator"],
    "eat":         ["kitchen", "dining table"],
    "food":        ["kitchen", "refrigerator", "dining table"],
    "thirsty":     ["kitchen", "refrigerator"],
    "drink":       ["kitchen", "refrigerator"],
    "coffee":      ["kitchen", "coffee machine"],
    # Work / study
    "work":        ["desk", "computer monitor", "laptop computer", "chair"],
    "study":       ["desk", "bookshelf", "laptop computer"],
    "homework":    ["desk", "bookshelf", "chair"],
    "type":        ["desk", "laptop computer", "keyboard"],
    "code":        ["desk", "computer monitor", "laptop computer"],
    # Reading
    "read":        ["bookshelf", "desk", "chair", "sofa"],
    "book":        ["bookshelf"],
    # Sitting / relaxing
    "sit":         ["chair", "sofa"],
    "relax":       ["sofa", "chair", "bed"],
    "chill":       ["sofa", "TV", "chair"],
    # Hygiene / health
    "wash":        ["sink", "bathroom"],
    "brush":       ["sink", "bathroom"],
    "shower":      ["bathroom"],
    "toilet":      ["toilet", "bathroom"],
    "sick":        ["bed", "sofa"],
    # Exercise
    "exercise":    ["gym equipment", "yoga mat", "chair"],
    "workout":     ["gym equipment", "yoga mat"],
    # Thinking / writing
    "write":       ["desk", "chair"],
    "think":       ["desk", "chair", "sofa"],
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

    _INTENT_EXTRACTION_PROMPT = (
        "You are a robotic assistant helping a user navigate to physical objects.\n"
        "The user expresses a need, feeling, or activity. Your job is to infer ALL physical\n"
        "objects or locations they might want to navigate to, ordered from most to least likely.\n"
        "Rules:\n"
        "  - Map feelings/needs to objects: \n"
        "      'sleepy/tired/exhausted' → bed, pillow, sofa\n"
        "      'bored' → TV, radio, bookshelf, sofa\n"
        "      'hungry/starving' → kitchen, dining table, refrigerator\n"
        "      'thirsty' → kitchen, refrigerator\n"
        "      'need to work/study' → desk, laptop computer, computer monitor\n"
        "      'want to read' → bookshelf, desk, chair\n"
        "      'want to relax/chill' → sofa, TV, chair\n"
        "      'need to wash/hygiene' → sink, bathroom\n"
        "  - Use generic category names (e.g. 'sofa' not 'large couch').\n"
        "  - Return a JSON array of strings, most likely first.\n"
        "  - Maximum 4 items. Minimum 1.\n"
        "  - Reply ONLY with the JSON array. No explanation.\n"
        "Examples:\n"
        "  'I am so sleepy' → [\"bed\", \"pillow\", \"sofa\"]\n"
        "  'I am bored' → [\"TV\", \"radio\", \"bookshelf\"]\n"
        "  'I need to eat something' → [\"kitchen\", \"dining table\", \"refrigerator\"]\n"
        "  'Can you show me the bathroom?' → [\"bathroom\"]\n"
        "  'I dropped my laptop' → [\"laptop computer\"]\n"
    )

    def _infer_object_queries(self, message: str, history: List[Dict]) -> List[str]:
        """Return a ranked list of candidate object queries for the user message.

        Priority:
        1. Direct keyword match in _IMPLICIT_OBJECT_MAP (returns the pre-defined list).
        2. QueryParser synonym normalisation for explicit single-object messages.
        3. LLM extraction via focused JSON-array intent-extraction prompt.
        """
        import json as _json
        import re as _re

        # 1. Keyword fast-path — return the full ranked list for the first matched keyword
        words = message.lower().split()
        for word in words:
            if word in _IMPLICIT_OBJECT_MAP:
                raw_list = _IMPLICIT_OBJECT_MAP[word]
                # Apply synonym normalisation on each item
                return [
                    QueryParser._OBJ_SYNONYMS.get(o.lower(), o) for o in raw_list
                ]

        # 2. Explicit object after stripping navigation prefix
        stripped = QueryParser._PREFIX_RE.sub("", message.strip(), count=1).strip().lower()
        if stripped:
            canonical = QueryParser._OBJ_SYNONYMS.get(stripped)
            if canonical:
                return [canonical]

        # 3. LLM extraction — returns a JSON list
        messages = [
            {"role": "system", "content": self._INTENT_EXTRACTION_PROMPT},
        ]
        for turn in history[-2:]:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": message})
        try:
            r = send_query(self._client, messages, self._model, temperature=0.0)
            raw = r.choices[0].message.content.strip()
            # Strip markdown fences
            raw = _re.sub(r"```(?:json)?\s*|\s*```", "", raw)
            parsed = _json.loads(raw)
            if isinstance(parsed, list) and parsed:
                return [
                    QueryParser._OBJ_SYNONYMS.get(str(o).lower(), str(o))
                    for o in parsed
                    if o
                ]
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("_infer_object_queries LLM call failed: %s", exc)

        # Fallback: use the stripped message
        fallback = stripped or message.strip()
        return [QueryParser._OBJ_SYNONYMS.get(fallback.lower(), fallback)]

    def _query_scene_graph(
        self, object_queries: "List[str] | str", top_k: int = 5
    ) -> List[Dict]:
        """Query the scene graph for one or more object queries and merge results.

        When multiple queries are given (implicit intent), each is searched in
        order and results are deduplicated by ``object_id``.  The ranking of the
        first query is preserved; results from later queries are appended only
        if their object has not already been retrieved.
        """
        if self.scene_graph is None:
            return []

        if isinstance(object_queries, str):
            object_queries = [object_queries]

        import numpy as np
        _T_SWITCH = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float
        )
        _T_TO_MAP = np.linalg.inv(_T_SWITCH)

        seen_ids: set = set()
        all_candidates: List[Dict] = []

        for query in object_queries:
            try:
                _floor, rooms, objects, _res = self.scene_graph.query_hierarchy(
                    query, top_k=top_k
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).error(
                    "Scene graph query failed for '%s': %s", query, e
                )
                continue

            for obj, room in zip(objects, rooms):
                oid = obj.object_id
                if oid in seen_ids:
                    continue
                seen_ids.add(oid)

                center_sg = np.array(obj.pcd.get_center())
                center_map = (_T_TO_MAP @ np.hstack((center_sg, 1.0)))[:3]

                floor_label = "?"
                obj_id_parts = str(oid).split("_")
                if obj_id_parts and self.scene_graph.floors:
                    idx = int(obj_id_parts[0]) if obj_id_parts[0].isdigit() else -1
                    if 0 <= idx < len(self.scene_graph.floors):
                        fl = self.scene_graph.floors[idx]
                        floor_label = fl.name or f"Floor {idx}"

                room_label = room.name or room.room_id
                all_candidates.append(
                    {
                        "name": obj.name,
                        "query": query,          # which sub-query found this object
                        "room": room_label,
                        "floor": floor_label,
                        "x": float(center_map[0]),
                        "y": float(center_map[1]),
                        "z": float(center_map[2]),
                        "object_id": oid,
                    }
                )

        return all_candidates

    def _start_navigation(
        self,
        message: str,
        history: List[Dict],
        nav_state: Dict,
    ) -> Tuple[str, Optional[Dict]]:
        """Begin a navigation dialogue from a fresh user message."""
        object_queries = self._infer_object_queries(message, history)
        nav_state["object_queries"] = object_queries
        nav_state["object_query"] = object_queries[0] if object_queries else message.strip()
        candidates = self._query_scene_graph(object_queries)
        nav_state["candidates"] = candidates

        if not candidates:
            nav_state.clear()
            response = self._general_chat(
                message + "\n(Note: I couldn't find any matching objects in my scene map. "
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
                f"I found a {c['name']} in {c['room']}, {c['floor']}. Shall I take you there?"
            )
            return response, None

        # Check if all candidates are in the same room — skip clarification
        rooms_seen = {(c["room"], c["floor"]) for c in candidates}
        if len(rooms_seen) == 1:
            c = candidates[0]
            nav_state["state"] = _NAV_STATE_CONFIRM
            nav_state["selected"] = c
            response = (
                f"I found a {c['name']} in {c['room']}, {c['floor']}. Shall I take you there?"
            )
            return response, None

        # Multiple locations — ask user to pick
        nav_state["state"] = _NAV_STATE_CLARIFY
        response = self._rephrase_candidates(candidates)
        return response, None

    def _rephrase_candidates(self, candidates: List[Dict]) -> str:
        """Ask the LLM to phrase the candidate list as a natural clarification question.

        Falls back to a plain numbered list if the LLM call fails.
        """
        # Build a compact structured summary for the LLM
        groups: dict = {}
        for c in candidates:
            key = (c.get("floor", ""), c.get("room", ""))
            groups.setdefault(key, []).append(c["name"])

        lines = []
        for (floor, room), names in groups.items():
            unique = sorted(set(names))
            label = f"{', '.join(unique)} in {room}, {floor}"
            lines.append(label)
        structured = "; ".join(lines)

        prompt = (
            "You are Motion, a friendly robot assistant. "
            "You found multiple objects the user might want to navigate to. "
            "Rephrase the following structured list into a single warm, natural sentence "
            "that groups objects by location and ends with a polite question asking which one they prefer. "
            "Do not use bullet points or numbers — write plain prose.\n\n"
            f"Candidates: {structured}"
        )
        try:
            resp = send_query(
                self._client,
                [{"role": "user", "content": prompt}],
                self._model,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            import logging

            logging.getLogger(__name__).warning("_rephrase_candidates LLM call failed: %s", e)
            # Plain fallback
            options_text = "\n".join(
                f"  {i + 1}. {c['name']} — {c['room']}, {c['floor']}"
                for i, c in enumerate(candidates)
            )
            return (
                f"I found {len(candidates)} options:\n{options_text}\n"
                "Which one would you like to go to?"
            )

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
                return (
                    "Understood, I won't navigate there. Let me know if you need anything else!",
                    None,
                )

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
        affirmatives = {
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
        }
        msg_lower = message.strip().lower()
        return any(a in msg_lower for a in affirmatives)

    def _resolve_selection(self, message: str, candidates: List[Dict]) -> Optional[Dict]:
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
