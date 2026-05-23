"""Interactive scene graph query selector test with mocked responses.

Tests the full post-intent conversation workflow: suggest → classify → select.
Uses fake scene graph candidates to exercise the same node functions as production.

Usage (from the fsr_vln directory):
    conda run -n fsrvln python tests/llm_selector.py
    conda run -n fsrvln python tests/llm_selector.py \\
        --room-labels "living room,bedroom,kitchen,office,bathroom" \\
        --object-labels "sofa,bed,sink,desk,TV,chair,table,refrigerator"

Type 'q' or Ctrl-C to quit.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

_DEFAULT_ROOM_LABELS = [
    "living room",
    "bedroom",
    "kitchen",
    "office",
    "bathroom",
    "dining room",
    "hallway",
    "storage",
    "entrance",
]
_DEFAULT_OBJECT_LABELS = [
    "sofa",
    "bed",
    "chair",
    "desk",
    "table",
    "sink",
    "TV",
    "refrigerator",
    "wardrobe",
    "bookshelf",
    "laptop computer",
    "computer monitor",
    "coffee table",
    "dining table",
    "trash can",
]

# Fake multi-room scene graph results for common objects
_FAKE_CANDIDATES = {
    "sofa": [("living room", "0"), ("bedroom", "1")],
    "chair": [("office", "0"), ("dining room", "0"), ("living room", "1")],
    "bed": [("bedroom", "0"), ("guest room", "1")],
    "desk": [("office", "0"), ("bedroom", "1")],
    "table": [("dining room", "0"), ("living room", "1")],
}


def _parse_label_arg(value: Optional[str], default: List[str]) -> List[str]:
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def _make_fake_candidates(obj: str, room: Optional[str] = None) -> List[dict]:
    """Return fake scene graph candidates matching real query result shape."""
    rooms = _FAKE_CANDIDATES.get(obj.lower(), [(room or "living room", "0")])
    return [
        {"name": f"{obj}_{i}", "room": r, "floor": f, "x": 10.0 + i, "y": 20.0, "z": 0.0}
        for i, (r, f) in enumerate(rooms)
    ]


def run_repl(room_labels: List[str], object_labels: List[str], floor_labels: List[str]) -> None:
    from memory.hmsg.tools.agents import (
        PHASE_IDLE,
        PHASE_NAV_SELECT,
        PHASE_SUGGEST,
        NavigationAgent,
        _format_candidates,
        nav_selector_node,
        suggest_classify_node,
        suggest_emit_node,
    )

    agent = NavigationAgent(
        scene_graph=None,
        room_types=room_labels,
        object_labels=object_labels,
        floor_labels=floor_labels,
    )

    print("\nRoom labels  :", room_labels, flush=True)
    print("Object labels:", object_labels, flush=True)
    print("Floor labels :", floor_labels, flush=True)
    print("\n[selector mode — mocked scene graph, real agent nodes]", flush=True)
    print("Type 'q' to quit.\n", flush=True)

    # Shared state dict — updated in-place each turn, mirrors real agent state
    state = {
        "llm_enable": True,
        "slow_reasoning": False,
        "room_labels": room_labels,
        "object_labels": object_labels,
        "floor_labels": floor_labels,
        "scene_graph": None,
        "messages": [],
        "phase": PHASE_IDLE,
        "goal": {},
        "alternatives": [],
        "candidates": [],
        "response": None,
        "action": None,
    }

    while True:
        try:
            prompt = "You: " if state["phase"] == PHASE_IDLE else "You (reply): "
            print(prompt, end="", flush=True)
            raw = sys.stdin.readline()
            if not raw:
                print()
                break
            raw = raw.rstrip("\n").strip()
        except KeyboardInterrupt:
            print()
            break

        if not raw or raw.lower() == "q":
            break

        state["user_message"] = raw
        t0 = time.perf_counter()

        try:
            phase = state["phase"]

            if phase == PHASE_IDLE:
                # --- intent extraction ---
                result = agent.extract_intent(raw, history=state["messages"], llm_enable=True)
                elapsed = time.perf_counter() - t0
                state["messages"].append({"role": "user", "content": raw})
                print(f"  extracted: {result}  ({elapsed:.2f}s)", flush=True)

                if result.get("intent") == "implicit":
                    state.update(
                        {"alternatives": result.get("alternatives", []), "phase": PHASE_SUGGEST}
                    )
                    out = suggest_emit_node(state)
                    state.update(out)
                    print(f"  Agent: {state['response']}", flush=True)

                elif result.get("object"):
                    obj = result["object"]
                    state["goal"] = {
                        "object": obj,
                        "room": result.get("room"),
                        "floor": result.get("floor"),
                    }
                    # Inject fake candidates instead of querying real scene graph
                    candidates = _make_fake_candidates(obj, result.get("room"))
                    state["candidates"] = candidates
                    print(
                        f"  [fake scene graph: {len(candidates)} result(s) for '{obj}']",
                        flush=True,
                    )

                    if len(candidates) == 1:
                        c = candidates[0]
                        msg = f"Navigate to {c['name']} in {c['room']} (floor {c['floor']}). Confirm?"
                        state["messages"].append({"role": "assistant", "content": msg})
                        state["phase"] = PHASE_NAV_SELECT
                        print(f"  Agent: {msg}", flush=True)
                    else:
                        msg = _format_candidates(obj, candidates)
                        state["messages"].append({"role": "assistant", "content": msg})
                        state["phase"] = PHASE_NAV_SELECT
                        print(f"  Agent: {msg}", flush=True)
                else:
                    print("  Agent: I couldn't identify an object. Try again.", flush=True)

            elif phase == PHASE_SUGGEST:
                # --- suggest_classify: resolve implicit intent ---
                out = suggest_classify_node(state)
                elapsed = time.perf_counter() - t0
                state.update(out)
                print(
                    f"  classify: type={out.get('goal', {}).get('object') or 'new_intent'}  ({elapsed:.2f}s)",
                    flush=True,
                )

                if state.get("response"):
                    # new_intent: node already set FALLBACK response
                    print(f"  Agent: {state['response']}", flush=True)
                elif state.get("goal", {}).get("object"):
                    obj = state["goal"]["object"]
                    candidates = _make_fake_candidates(obj)
                    state["candidates"] = candidates
                    print(
                        f"  [fake scene graph: {len(candidates)} result(s) for '{obj}']",
                        flush=True,
                    )
                    msg = (
                        _format_candidates(obj, candidates)
                        if len(candidates) > 1
                        else (
                            f"Navigate to {candidates[0]['name']} in {candidates[0]['room']}. Confirm?"
                        )
                    )
                    state["messages"].append({"role": "assistant", "content": msg})
                    state["phase"] = PHASE_NAV_SELECT
                    print(f"  Agent: {msg}", flush=True)

            elif phase == PHASE_NAV_SELECT:
                # --- nav_selector: pick from candidates ---
                out = nav_selector_node(state)
                elapsed = time.perf_counter() - t0
                state.update(out)
                print(f"  select: action={state.get('action')}  ({elapsed:.2f}s)", flush=True)
                print(f"  Agent: {state['response']}", flush=True)

        except Exception as exc:
            print(f"  [error] {exc}\n", flush=True)
            state["phase"] = PHASE_IDLE

        print(flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive selector test with mocked scene graph")
    ap.add_argument("--room-labels", default=None, metavar="CSV")
    ap.add_argument("--object-labels", default=None, metavar="CSV")
    ap.add_argument("--floor-labels", default=None, metavar="CSV")
    args = ap.parse_args()

    room_labels = _parse_label_arg(args.room_labels, _DEFAULT_ROOM_LABELS)
    object_labels = _parse_label_arg(args.object_labels, _DEFAULT_OBJECT_LABELS)
    floor_labels = _parse_label_arg(args.floor_labels, ["0", "1"])

    from memory.hmsg.tools.llm import check_llm_connection

    ok, reason = check_llm_connection(timeout=5.0)
    if not ok:
        print(f"[!] LLM unreachable: {reason}", file=sys.stderr)
        sys.exit(1)

    run_repl(room_labels, object_labels, floor_labels)


if __name__ == "__main__":
    main()
