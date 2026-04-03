"""LLM utilities for query parsing and inference with multi-provider support."""

import os
from functools import lru_cache
from typing import List, Optional, Tuple

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


if __name__ == "__main__":
    # Test example
    parser = QueryParser()
    result = parser.parse("sofa in living room on floor 1")
    print(f"Result: {result}")
