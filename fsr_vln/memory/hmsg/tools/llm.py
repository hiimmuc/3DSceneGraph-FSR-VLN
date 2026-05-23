"""LLM utilities for query parsing and inference via OpenAI-compatible APIs."""

import json
import logging
import os
import re
import socket
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
)

load_dotenv()

_log = logging.getLogger(__name__)

from memory.hmsg.tools.prompts import (  # noqa: E402
    FLOOR_INFER_SYSTEM,
    QUERY_PARSER_SYSTEM,
    ROOM_INFER_SYSTEM,
)

# Default endpoint / model — set LLM_ENDPOINT / LLM_MODEL env vars to override.
_DEFAULT_ENDPOINT = "http://localhost:8000/v1"
_DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def create_llm_client(provider: Optional[str] = None) -> Tuple[object, str]:
    """Create LLM client and return (client, model_name) tuple.

    Args:
        provider: Provider label used for logging only.
                  If None, reads LLM_PROVIDER env var (default: "openai-compatible").

    Returns:
        Tuple of (client, model_name).
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "openai-compatible")

    base_url = os.environ.get("HTTP_ENDPOINT", _DEFAULT_ENDPOINT)
    model = os.environ.get("MODEL_NAME", _DEFAULT_MODEL)
    api_key = os.environ.get("API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY"

    _log.info(
        "Initializing LLM client: provider=%s endpoint=%s model=%s",
        provider,
        base_url,
        model,
    )
    return OpenAI(base_url=base_url, api_key=api_key, timeout=15), model


def send_query(
    client: object, messages: List[dict], model: str, temperature: float = 0.0, **kwargs
) -> object:
    """Send a text query to the LLM and return the raw API response."""
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Module-level singleton client (created once, reused for all text queries)
# ---------------------------------------------------------------------------

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
    """Send a cached query to avoid redundant LLM calls for identical inputs.

    Args:
        model: Model name.
        messages_str: JSON-serialized messages list (used as cache key).
        temperature: Must be 0.0 for deterministic caching.

    Returns:
        Response content string.
    """
    if temperature != 0.0:
        raise ValueError("Caching only works with temperature=0.0 for deterministic responses")
    client, _ = _get_cached_client()
    messages = json.loads(messages_str)
    response = send_query(client, messages, model, temperature=temperature)
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Query parser
# ---------------------------------------------------------------------------


class QueryParser:
    """Parse hierarchical queries into floor, room, and object components.

    Uses a structured JSON-output prompt so the LLM response is machine-readable
    without fragile comma-splitting, and applies synonym normalization before
    returning results.
    """

    _VALID_SPECS = frozenset(
        {
            ("obj", "room", "floor"),
            ("obj", "room"),
            ("obj", "floor"),
            ("obj",),
        }
    )

    _PREFIX_RE = re.compile(
        r"^(?:"
        r"find(?:\s+me)?|where(?:'s|\s+is)|locate|show(?:\s+me)?"
        r"|bring(?:\s+me)?|get(?:\s+me)?|take(?:\s+me\s+to)?"
        r"|navigate(?:\s+to)?|go(?:\s+to)?|i\s+need"
        r")[\s,]+(?:the\s+|a\s+|an\s+)?",
        re.IGNORECASE,
    )

    _SYSTEM_PROMPT = QUERY_PARSER_SYSTEM

    def __init__(self, client: object = None, model: str = None):
        if client is None or model is None:
            client, model = _get_cached_client()
        self.client = client
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_room(self, room: Optional[str]) -> Optional[str]:
        return room.strip().lower() if room else None

    def _normalize_object(self, obj: Optional[str]) -> Optional[str]:
        return obj.strip() if obj else None

    def _strip_prefix(self, instruction: str) -> str:
        stripped = self._PREFIX_RE.sub("", instruction.strip()).strip()
        return stripped or instruction.strip()

    def _parse_json_response(
        self, raw: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw.strip())
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            _log.warning("No JSON object found in LLM response: %r", raw)
            return None, None, None, []
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError as exc:
            _log.warning("JSON decode error (%s) in: %r", exc, m.group())
            return None, None, None, []

        def _clean(v: Any) -> Optional[str]:
            if v is None:
                return None
            s = str(v).strip()
            return None if s.lower() == "null" or s == "" else s

        floor = _clean(data.get("floor"))
        room = _clean(data.get("room"))
        obj = _clean(data.get("object"))

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
        """Parse instruction into (floor, room, object) hierarchy components."""
        floor, room, obj, _alts = self.parse_with_alts(instruction, spec)
        return floor, room, obj

    def parse_with_alts(
        self, instruction: str, spec: Tuple[str, ...] = ("obj", "room", "floor")
    ) -> Tuple[Optional[str], Optional[str], Optional[str], List[str]]:
        """Parse instruction and return (floor, room, primary_object, alt_objects).

        ``alt_objects`` is a ranked list of alternative object categories for
        implicit-intent queries (e.g. 'I am sleepy' → alts=['pillow', 'sofa']).
        """
        if spec not in self._VALID_SPECS:
            raise ValueError(f"Unknown query spec: {spec}. Available: {list(self._VALID_SPECS)}")

        if spec == ("obj",):
            return None, None, self._normalize_object(self._strip_prefix(instruction)), []

        messages = [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": instruction.strip()},
        ]
        raw_result = send_query_cached(
            self.model, json.dumps(messages, ensure_ascii=False), temperature=0.0
        )
        _log.debug("LLM parsed %r -> %r", instruction, raw_result)

        floor, room, obj, alt_objects = self._parse_json_response(raw_result)

        floor = floor.strip() if floor else None
        room = self._normalize_room(room)
        obj = self._normalize_object(obj) or self._strip_prefix(instruction)
        alt_objects = [self._normalize_object(a) for a in alt_objects if a]

        _log.debug("  -> floor=%r  room=%r  obj=%r  alts=%s", floor, room, obj, alt_objects)
        return floor, room, obj, alt_objects


# ---------------------------------------------------------------------------
# Standalone inference helpers
# ---------------------------------------------------------------------------


def infer_room_type_from_objects(
    object_list: List[str], candidate_room_types: List[str] = None
) -> str:
    """Infer room type from a list of objects using the cached LLM client."""
    client, model = _get_cached_client()
    constraint = f"Choose from: {', '.join(candidate_room_types)}." if candidate_room_types else ""
    messages = [
        {"role": "system", "content": ROOM_INFER_SYSTEM},
        {"role": "user", "content": "Objects: bed, wardrobe, chair. Room type?"},
        {"role": "assistant", "content": "Bedroom"},
        {"role": "user", "content": f"Objects: {', '.join(object_list)}. {constraint} Room type?"},
    ]
    room_type = send_query(client, messages, model).choices[0].message.content.strip()
    _log.debug("Inferred room type: %s", room_type)
    return room_type


def check_llm_connection(timeout: float = 5.0) -> Tuple[bool, str]:
    """Probe the configured OpenAI-compatible endpoint and model availability.

    Returns:
        ``(True, "")`` when reachable; ``(False, reason)`` otherwise.
    """
    provider = os.environ.get("PROVIDER", "openai-compatible")
    base_url = os.environ.get("HTTP_ENDPOINT", _DEFAULT_ENDPOINT)
    model_name = os.environ.get("MODEL_NAME", _DEFAULT_MODEL)
    _log.debug(
        "Checking LLM connection: provider=%s endpoint=%s model=%s",
        provider,
        base_url,
        model_name,
    )

    try:
        client, model = create_llm_client()
    except Exception as exc:
        return False, f"Client creation failed: {exc}"

    # Preferred probe: model list endpoint (low-cost and provider-agnostic).
    try:
        models = client.with_options(timeout=timeout, max_retries=0).models.list()
        available = [getattr(m, "id", "") for m in getattr(models, "data", [])]
        if available and not any(model_name in m for m in available):
            _log.warning("Model %r not found in endpoint model list: %s", model_name, available)
        return True, ""
    except APIStatusError:
        return True, ""  # server responded (4xx/5xx) → reachable
    except Exception:
        pass

    # Fallback probe: minimal chat completion.
    try:
        client.with_options(timeout=timeout, max_retries=0).chat.completions.create(
            model=model, messages=[{"role": "user", "content": "ping"}], max_tokens=1
        )
        return True, ""
    except APIStatusError:
        return True, ""  # server responded (4xx/5xx) → reachable
    except APIConnectionError as exc:
        return False, f"Connection error: {exc}"
    except APITimeoutError as exc:
        return False, f"Timeout: {exc}"
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Backward-compatible wrapper functions
# ---------------------------------------------------------------------------


def infer_floor_id_from_query(floor_ids_list: List[int], query: str) -> int:
    """Infer floor ID from a natural-language query. Returns first floor on failure."""
    client, model = _get_cached_client()
    messages = [
        {"role": "system", "content": FLOOR_INFER_SYSTEM},
        {"role": "user", "content": f"Floors: {floor_ids_list}. Query: {query}. Which floor?"},
    ]
    result = (
        send_query(client, messages, model, temperature=0.0).choices[0].message.content.strip()
    )
    try:
        floor_id = int(result)
        if floor_id in floor_ids_list:
            return floor_id
    except ValueError:
        pass
    return floor_ids_list[0] if floor_ids_list else 1


def parse_hierarchy_query(
    cfg, instruction: str, parser: "QueryParser" = None
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse a natural language instruction into (floor, room, object) components."""
    if parser is None:
        parser = QueryParser()
    spec = tuple(cfg.main.long_query.spec) if hasattr(cfg, "main") else ("obj", "room", "floor")
    return parser.parse(instruction, spec)


def publish_navigation_goal(
    goal: Dict[str, Any],
    host: str = "127.0.0.1",
    port: int = 5005,
) -> bool:
    """Send a navigation goal to the ROS 2 goal bridge via UDP."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.sendto(json.dumps(goal).encode(), (host, port))
        return True
    except Exception as exc:
        _log.error("publish_navigation_goal failed: %s", exc)
        return False


if __name__ == "__main__":
    parser = QueryParser()
    result = parser.parse("sofa in living room on floor 1")
    print(f"Result: {result}")
