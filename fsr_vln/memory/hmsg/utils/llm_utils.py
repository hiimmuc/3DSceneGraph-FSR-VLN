"""Backward-compatible shim — implementation lives in memory.hmsg.tools.llm."""

from memory.hmsg.tools.llm import (  # noqa: F401
    check_llm_connection,
    infer_room_type_from_objects,
    publish_navigation_goal,
)
