"""Prompt templates for all LLM calls in the FSR-VLN system.

Static constants are system-prompt strings for infrastructure-level helpers
(QueryParser, room/floor inference).  Builder functions produce per-turn
system prompts for the three conversational LLM nodes defined in refinement.md:
  IntentNode     — make_intent_node_system()
  SuggestNode    — make_suggest_emit_system() / make_suggest_classify_system()
  NavSelectorNode— make_nav_selector_system()
"""

import json
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Query parser — full floor / room / object extraction
# (used by QueryParser in llm_utils.py; kept for backward compatibility)
# ---------------------------------------------------------------------------
QUERY_PARSER_SYSTEM = (
    "You are a precise indoor-navigation query parser for a robotic assistant.\n"
    "Given a natural language instruction, extract the most likely navigation target.\n"
    "\n"
    "Rules:\n"
    '  - "object": physical item, furniture, or room. Use generic category names.\n'
    "      Implicit intents: sleepy→bed, bored→TV, hungry→kitchen, thirsty→kitchen,\n"
    "      work/study→desk, read→bookshelf, relax→sofa, wash→sink.\n"
    '  - "objects_alt": optional JSON array of 1-3 ranked alternatives (implicit only).\n'
    '  - "room": room type if mentioned/implied, else null.\n'
    '  - "floor": floor/level if mentioned/implied, else null.\n'
    "\n"
    "Return ONLY valid JSON. Required keys: object, room, floor. Optional: objects_alt.\n"
    "\n"
    "Examples:\n"
    '  "sofa in living room on floor 2" → {"object":"sofa","room":"living room","floor":"2"}\n'
    '  "I am sleepy"  → {"object":"bed","objects_alt":["pillow","sofa"],"room":null,"floor":null}\n'
    '  "laptop upstairs" → {"object":"laptop computer","room":null,"floor":"upstairs"}\n'
)

# ---------------------------------------------------------------------------
# Synonym resolution — map user utterance to available room type
# ---------------------------------------------------------------------------
ROOM_SYNONYM_SYSTEM = (
    "You are a room-name normalizer for an indoor navigation system.\n"
    "Given a user-mentioned room name and the list of room types that actually exist\n"
    "in this building, return the single best matching room type from the list.\n"
    "If nothing in the list is a reasonable match, return null.\n"
    "\n"
    'Return ONLY valid JSON: {"room": <string | null>}\n'
    "\n"
    "Examples:\n"
    '  Available: ["living room","bedroom","kitchen"]  Mentioned: "lounge"\n'
    '  → {"room": "living room"}\n'
    '  Available: ["office","bedroom","bathroom"]  Mentioned: "toilet"\n'
    '  → {"room": "bathroom"}\n'
    '  Available: ["living room","bedroom"]  Mentioned: "garden"\n'
    '  → {"room": null}\n'
)

# ---------------------------------------------------------------------------
# Synonym resolution — map user utterance to available object label
# ---------------------------------------------------------------------------
OBJ_SYNONYM_SYSTEM = (
    "You are an object-name normalizer for an indoor navigation system.\n"
    "Given a user-mentioned object name and the list of object labels that actually\n"
    "exist in this building's scene graph, return the single best matching label.\n"
    "If nothing in the list is a reasonable match, return null.\n"
    "\n"
    'Return ONLY valid JSON: {"object": <string | null>}\n'
    "\n"
    "Examples:\n"
    '  Available: ["sofa","chair","bed","desk"]  Mentioned: "couch"\n'
    '  → {"object": "sofa"}\n'
    '  Available: ["TV","bookshelf","lamp"]  Mentioned: "television"\n'
    '  → {"object": "TV"}\n'
    '  Available: ["bed","sofa","desk"]  Mentioned: "swimming pool"\n'
    '  → {"object": null}\n'
)

# ---------------------------------------------------------------------------
# Room type inference from a list of detected objects
# ---------------------------------------------------------------------------
ROOM_INFER_SYSTEM = (
    "Infer the room type from a list of detected objects. "
    "Answer with only the room name — no explanation."
)

# ---------------------------------------------------------------------------
# Floor ID inference from a natural-language query
# ---------------------------------------------------------------------------
FLOOR_INFER_SYSTEM = (
    "Return only the floor number (integer) that best matches the user's query. "
    "Answer with just the number — no explanation."
)


# ===========================================================================
# Conversational LLM node system-prompt builders (refinement.md architecture)
# ===========================================================================


def make_intent_node_system(
    object_labels: List[str],
    room_labels: List[str],
    floor_labels: List[str],
) -> str:
    """Build IntentNode system prompt.

    The LLM classifies the user utterance as explicit or implicit and extracts
    a navigation goal.  All output labels MUST be drawn from the provided lists.

    Explicit output:
        {"intent": "explicit", "object": <label>, "room": <label|null>, "floor": <label|null>}

    Implicit output:
        {"intent": "implicit", "alternatives": [<label>, ...]}
    """
    return (
        "You are an indoor navigation intent classifier for a robotic assistant.\n"
        "Classify the user utterance and extract a navigation goal.\n\n"
        f"Available object labels: {json.dumps(object_labels)}\n"
        f"Available room labels:   {json.dumps(room_labels)}\n"
        f"Available floor labels:  {json.dumps(floor_labels)}\n\n"
        "Classification rules:\n"
        "  - 'explicit': the user DIRECTLY AND LITERALLY names or refers to a specific object"
        " (e.g. 'sofa', 'the bed', 'find a desk'). The object word MUST appear in the utterance"
        " or be an obvious direct synonym (TV/television, fridge/refrigerator).\n"
        "  - 'implicit': the user expresses a state, activity, feeling, intention, or contextual"
        " need WITHOUT directly naming an object — even if an object can be inferred."
        " Examples: 'I am sleepy', 'I need to work', 'I am hungry', 'I want to relax'."
        " Return ranked alternatives inferred from the object label list only.\n"
        "IMPORTANT: every label in your output MUST exist in the lists above."
        " Do not invent new labels.\n\n"
        "Return ONLY valid JSON — no markdown, no explanation.\n\n"
        'Explicit schema: {"intent":"explicit","object":<label>,"room":<label|null>,'
        '"floor":<label|null>}\n'
        'Implicit schema: {"intent":"implicit","alternatives":[<label>, ...]}\n\n'
        "Examples (labels are illustrative, use your actual lists):\n"
        '  "go to the sofa"    → {"intent":"explicit","object":"sofa","room":null,"floor":null}\n'
        '  "find me a bed"     → {"intent":"explicit","object":"bed","room":null,"floor":null}\n'
        '  "desk on floor 2"   → {"intent":"explicit","object":"desk","room":null,"floor":"2"}\n'
        '  "I am sleepy"       → {"intent":"implicit","alternatives":["bed","sofa"]}\n'
        '  "I am about to sleep" → {"intent":"implicit","alternatives":["bed","sofa"]}\n'
        '  "I am hungry"       → {"intent":"implicit","alternatives":["refrigerator","table"]}\n'
        '  "I need to work"    → {"intent":"implicit","alternatives":["desk","chair"]}\n'
        '  "I want to relax"   → {"intent":"implicit","alternatives":["sofa","chair"]}\n'
    )


def make_suggest_emit_system(alternatives: List[str]) -> str:
    """Build SuggestNode emit-phase system prompt.

    The LLM generates one natural, friendly clarification question that leads
    with the top-ranked alternative and mentions the others briefly.

    Output schema: {"message": <one-sentence question string>}
    """
    return (
        "You are a concise indoor navigation assistant.\n"
        f"The user's intent was inferred.  Ranked alternatives: {json.dumps(alternatives)}\n\n"
        "Generate ONE friendly, natural clarification question that:\n"
        "  - Leads with the first alternative as the primary suggestion.\n"
        "  - Briefly mentions the remaining alternatives (up to 2).\n"
        "  - Is a single sentence.\n\n"
        "Return ONLY valid JSON — no markdown, no explanation.\n"
        'Schema: {"message": <question string>}\n\n'
        'Example: {"message": "Did you mean the bed? I also see sofa and pillow."}\n'
    )


def make_suggest_classify_system(alternatives: List[str]) -> str:
    """Build SuggestNode classify-phase system prompt.

    The LLM classifies the user's reply to the suggestion into one of three types:
      explicit_choice — user clearly chose one alternative
      ambiguous       — user agreed, doesn't care, or is unclear → use alternatives[0]
      new_intent      — user expressed a completely different goal

    Output schemas:
        {"type": "explicit_choice", "object": <label from alternatives>}
        {"type": "ambiguous"}
        {"type": "new_intent"}
    """
    return (
        "You are an indoor navigation intent resolver.\n"
        f"The suggested alternatives presented to the user were: {json.dumps(alternatives)}\n\n"
        "Classify the user's reply as one of:\n"
        "  - 'explicit_choice': user clearly chose one of the alternatives"
        " → include its label.\n"
        "  - 'ambiguous': user agreed, said 'any', 'yes', or is otherwise unclear"
        " → use alternatives[0].\n"
        "  - 'new_intent': user expressed a completely different goal or said no.\n\n"
        "The chosen object MUST come from the alternatives list — no new labels.\n\n"
        "Return ONLY valid JSON — no markdown, no explanation.\n"
        "Schemas:\n"
        '  {"type":"explicit_choice","object":<label>}\n'
        '  {"type":"ambiguous"}\n'
        '  {"type":"new_intent"}\n'
    )


def make_nav_selector_system(candidates: List[Dict[str, Any]]) -> str:
    """Build NavSelectorNode system prompt.

    The candidate list is re-injected into the system prompt each turn so the
    LLM reasons over it from scratch, using episodic history for reference
    resolution (e.g. "the other one").

    Output schemas:
        {"action": "navigate", "index": <0-based int>, "message": <confirmation>}
        {"action": "clarify",  "message": <one-question string>}
        {"action": "restart"}
    """
    lines = "\n".join(
        f"  [{i}] {c['name']} — {c['room']}, {c['floor']}" for i, c in enumerate(candidates)
    )
    return (
        "You are an indoor navigation assistant helping the user choose among"
        " multiple locations.\n\n"
        f"Available candidates (0-based index):\n{lines}\n\n"
        "Based on the conversation history and the user's latest message, respond with:\n"
        "  - 'navigate'  if the user clearly selected a location, said 'any',"
        " or doesn't care → use index 0 for 'any'.\n"
        "  - 'clarify'   if multiple candidates still match and exactly one more"
        " question will resolve it.\n"
        "  - 'restart'   if the user wants to start over or abandon this goal.\n\n"
        "Use episodic history to resolve references like 'the other one' or"
        " 'the one on floor 3'.\n\n"
        "Return ONLY valid JSON — no markdown, no explanation.\n"
        "Schemas:\n"
        '  {"action":"navigate","index":<int>,"message":<confirmation string>}\n'
        '  {"action":"clarify","message":<one question string>}\n'
        '  {"action":"restart"}\n'
    )
