"""Interactive LLM retrieval conversation test.

Uses NavigationAgent (same path as the main application) with user-supplied
room and object labels so the LLM reasons only against the provided vocabulary.

Usage (from the fsr_vln directory):
    conda run -n fsrvln python tests/llm_retrieval.py
    conda run -n fsrvln python tests/llm_retrieval.py \\
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

# Default labels used when none are supplied via CLI
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


def _parse_label_arg(value: Optional[str], default: List[str]) -> List[str]:
    if not value:
        return default
    return [v.strip() for v in value.split(",") if v.strip()]


def run_repl(
    room_labels: List[str],
    object_labels: List[str],
    floor_labels: List[str],
) -> None:
    from memory.hmsg.tools.agents import NavigationAgent, _suggest_template

    agent = NavigationAgent(
        scene_graph=None,
        room_types=room_labels,
        object_labels=object_labels,
        floor_labels=floor_labels,
    )

    print("\nRoom labels  :", room_labels, flush=True)
    print("Object labels:", object_labels, flush=True)
    print("Floor labels :", floor_labels, flush=True)
    print("\n[extraction-only mode — no scene-graph query]", flush=True)
    print("Type 'q' to quit.\n", flush=True)

    history: list = []

    while True:
        try:
            print("You: ", end="", flush=True)
            raw = sys.stdin.readline()
            if not raw:  # EOF
                print()
                break
            raw = raw.rstrip("\n").strip()
        except KeyboardInterrupt:
            print()
            break

        if not raw or raw.lower() == "q":
            break

        t0 = time.perf_counter()
        try:
            result = agent.extract_intent(raw, history=history, llm_enable=True)
        except Exception as exc:
            print(f"  [error] {exc}\n", flush=True)
            continue

        elapsed = time.perf_counter() - t0
        history.append({"role": "user", "content": raw})

        print(f"  extracted: {result}  ({elapsed:.2f}s)", flush=True)

        if result.get("intent") == "implicit":
            alts = result.get("alternatives", [])
            suggestion = _suggest_template(alts)
            history.append({"role": "assistant", "content": suggestion})
            print(f"  Agent: {suggestion}", flush=True)
        elif result.get("object"):
            msg = (
                f"Looking for {result['object']}"
                + (f" in {result['room']}" if result.get("room") else "")
                + (f" on floor {result['floor']}" if result.get("floor") else "")
                + "."
            )
            history.append({"role": "assistant", "content": msg})
            print(f"  Agent: {msg}", flush=True)

        print(flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive LLM retrieval conversation test")
    ap.add_argument(
        "--room-labels",
        default=None,
        metavar="CSV",
        help="Comma-separated room label list (default: built-in set).",
    )
    ap.add_argument(
        "--object-labels",
        default=None,
        metavar="CSV",
        help="Comma-separated object label list (default: built-in set).",
    )
    ap.add_argument(
        "--floor-labels",
        default=None,
        metavar="CSV",
        help="Comma-separated floor label list (default: '0,1').",
    )
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
