import os
import re
import time
from functools import lru_cache
from typing import List, Tuple

import openai
from openai import AzureOpenAI, OpenAI

# ---------------------------------------------------------------------------
# LLM provider factory
# Set LLM_PROVIDER=azure to use Azure OpenAI; defaults to Ollama (qwen2.5).
# Relevant env vars:
#   LLM_PROVIDER          : "ollama" (default) | "azure"
#   OLLAMA_BASE_URL       : default http://localhost:11434/v1
#   OLLAMA_MODEL          : default qwen2.5
#   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
#   AZURE_OPENAI_MODEL
# ---------------------------------------------------------------------------
_DEFAULT_OLLAMA_MODEL = "qwen3-vl:4b"
_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"


def create_llm_client(provider: str = None):
    """Return (client, model_name) for the configured LLM provider.

    The returned client exposes the same ``client.chat.completions.create``
    interface regardless of the backend, making it trivial to switch.

    Returns:
        ed client exposes the same ``client.chat.completions.create``
    interface regardless of the backend, making it trivial to switch."""
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "ollama")

    if provider == "azure":
        client = AzureOpenAI(
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", "xxxx"),
            api_key=os.environ.get("AZURE_OPENAI_API_KEY", "xxxx"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "xxxx"),
        )
        model = os.environ.get("AZURE_OPENAI_MODEL", "xxxx")
    else:  # ollama (default)
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        model = os.environ.get("OLLAMA_MODEL", _DEFAULT_OLLAMA_MODEL)
        client = OpenAI(base_url=base_url, api_key="ollama")

    return client, model


def infer_floor_id_from_query(floor_ids: List[int], query: str) -> int:
    """
    Return the floor id from the floor_ids_list that match with the query.

    Args:
        floor_ids (List[int]): a list starting from 1 to highest floor level
        query (str): a text description of the floor level number

    Returns:
        int: the target floor number (starting from 1)
    """
    floor_ids_str = [str(i) for i in floor_ids]
    floor_ids_str = ", ".join(floor_ids_str)

    openai_key = os.environ["OPENAI_KEY"]
    openai.api_key = openai_key
    question = f"""You are a floor detector. You can infer the floor number based on a query.

The query is: {query}.
The floor number list is: {floor_ids_str}.
Please answer the floor number in one integer."""
    print(question)
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=question,
        max_tokens=64,
        temperature=0.0,
        stop=None,
    )
    result = response["choices"][0]["text"]
    try:
        result = int(result)
    except BaseException:
        print(f"The return answer is not an integer. The answer is: {result}")
        assert False
    return result


def infer_room_type_from_object_list_chat(
    object_list: List[str], default_room_type: List[str] = None
) -> str:
    """
    Generate a room type based on a list of objects contained in the room with
    chat.

    Args:
        object_list (List[str]): a list of object names contained in the room
        default_room_type (List[str] = None): the inferred room type should be from this list

    Returns:
        str: a text describing the room type
    """

    client, gpt_model = create_llm_client()

    room_types = ""
    if default_room_type is not None:
        room_types = ", ".join(default_room_type)
        room_types = (
            "Please pick the most matching room type from the following list: " + room_types + "."
        )

    objects = ", ".join(object_list)
    # print(f"Objects list: {objects}")
    print(f"Room types: {room_types}")

    question = """"""
    print(question)
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "system",
                "content": "You are a room type detector. You can infer a room type based on a list of objects.",
            },
            {
                "role": "user",
                "content": "The list of objects contained in this room are: bed, wardrobe, chair, sofa. What is the room type? Please just answer the room name.",
            },
            {
                "role": "assistant",
                "content": "bedroom",
            },
            {
                "role": "user",
                "content": "The list of objects contained in this room are: tv, table, chair, sofa. Please pick the most matching room type from the following list: living room, bedroom, bathroom, kitchen. What is the room type? Please just answer the room name.",
            },
            {
                "role": "assistant",
                "content": "living room",
            },
            {
                "role": "user",
                "content": f"The list of objects contained in this room are: {objects}. {room_types} What is the room type? Please just answer the room name.",
            },
        ],
    )
    print(response)
    result = response.choices[0].message.content
    print("The room type is: ", result)
    return result


class Conversation:
    def __init__(self, messages: List[dict], include_env_messages: bool = False) -> None:
        """An interface to OPENAI chat API.

        Args:
            messages (List[dict]): The list of messages to be sent to the chat API
            include_env_messages (bool, optional): Boolean controlling if environment message is sent. Defaults to False.
        """
        self._messages = messages
        self._include_env_messages = include_env_messages

    def add_message(self, message: dict):
        self._messages.append(message)

    @property
    def messages(self):
        if self._include_env_messages:
            return self._messages
        else:
            return [m for m in self._messages if m["role"].lower() not in ["env", "environment"]]

    @property
    def messages_including_env(self):
        return self._messages


@lru_cache(maxsize=None)
def send_query_cached(client, messages: list, model: str, temperature: float):
    assert (
        temperature == 0.0
    ), "Caching only works for temperature=0.0, as eitherwise we want to get different responses back"
    messages = [dict(m) for m in messages]
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def send_query(client, messages: list, model: str, temperature: float):
    return client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )


def parse_hier_query(params, instruction: str) -> Tuple[str, str, str]:
    """Parse long language query into a list of short queries at floor, room, and object level

    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")"""

    client, gpt_model = create_llm_client()

    # Depending on the query spec, parse the query differently:
    if set(params.main.long_query.spec) == {"obj", "room", "floor"}:
        system_prompt = "You are a query parser. You have to parse a sentence into a floor, a room and an object."
        prompt = f"Please parse the following: {instruction}"
        prompt += "Output Response Format: Comma-separated list of these three things such as [floor 2, living room, couch]"
    elif set(params.main.long_query.spec) == {"obj", "room"}:
        system_prompt = (
            "You are a query parser. You have to parse a sentence into a room and an object."
        )
        prompt = f"Please parse the following: {instruction}"
        prompt += "Output Response Format: Comma-separated list of these two things such as [living room, couch]"
    elif set(params.main.long_query.spec) == {"obj", "floor"}:
        system_prompt = (
            "You are a query parser. You have to parse a sentence into a floor and an object."
        )
        prompt = f"Please parse the following: {instruction}"
        prompt += "Output Response Format: Comma-separated list of these two things such as [floor 2, couch]"
    elif set(params.main.long_query.spec) == {"obj"}:
        # return directly and not use the LLM for parsing
        print("floor, room, object:", None, None, instruction)
        return [None, None, instruction.strip()]

    conversation = Conversation(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    response = send_query(client, messages=conversation.messages, model=gpt_model, temperature=0.0)
    raw_result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")

    # Split safely
    spec = set(params.main.long_query.spec)
    parts = [x.strip() for x in raw_result.split(",")]
    # Ensure always 3 elements
    floor, room, obj = None, None, None
    try:
        if spec == {"obj", "room", "floor"}:
            floor, room, obj = (parts + [None] * 3)[:3]
        elif spec == {"obj", "room"}:
            room, obj = (parts + [None] * 2)[:2]
        elif spec == {"obj", "floor"}:
            floor, obj = (parts + [None] * 2)[:2]
    except Exception as e:
        print(f"Warning: failed to parse LLM result: {raw_result}, error: {e}")
        floor, room, obj = None, None, instruction.strip()

    print("floor, room, object:", floor, room, obj)
    return (floor, room, obj)


def generate_clip_probes(self, instruction):

    prompt = f"""You are an AI assistant for visual navigation.

Given a navigation instruction, extract the main target object(s) mentioned or implied.
If the instruction does not explicitly mention an object, infer the most likely target object(s) based on common sense and the user's intent.
Generate a diverse bullet list of English phrases for CLIP-based image retrieval, including synonyms and descriptive variants.
If no clear object is mentioned, output an empty list.
Instruction: {instruction}"""
    response_flag = False
    while not response_flag:
        try:
            print("Sending request stage 1 ...")
            response = self.client.chat.completions.create(
                model=self.gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    }
                ],
                seed=123,
            )
            response_flag = True
        except Exception as e:
            print(e)
            time.sleep(1)
            print("Retrying ...")
    response = response.choices[0].message.content
    text_probes = re.findall(r"-(.*?)\n", response)
    text_probes = [item.strip(' "-') for item in text_probes]
    text_probes = [item for item in text_probes if len(item) > 0]
    return text_probes


def parse_hier_query_use_prompt_insentence_parse(params, instruction: str) -> Tuple[str, str, str]:
    """Parse long language query into a list of short queries at floor, room, and object level

    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")"""

    client, gpt_model = create_llm_client()

    # Depending on the query spec, parse the query differently:
    if set(params.main.long_query.spec) == {"obj", "room", "floor"}:
        system_prompt = "You are a query parser. Your task is to parse a sentence into floor, room, and object. If only a room or object can be parsed, leave the other field empty. Object descriptions must be in English; floor and room may be in Chinese."
        # system_prompt = "You are a query parser named Digua. Ignore all occurrences of the word 'Digua' in the sentence. Your task is to parse a sentence into floor, room, and object. If only a room or object can be parsed, leave the other field empty. Object descriptions must be in English; floor and room may be in Chinese."

        prompt = f"Please parse the following sentence: {instruction}"
        prompt += "Output format: a comma-separated list in the order of floor, room, and object. Example: [Floor 1, Digua Office Area, sofa]"
    elif set(params.main.long_query.spec) == {"obj", "room"}:
        system_prompt = "You are a query parser named Digua. Your task is to parse a sentence into room and object."
        prompt = f"Please parse the following sentence: {instruction}"
        prompt += "Output format: a comma-separated list in the order of room and object. Example: [Horizon Exhibition Hall, sofa]"
    elif set(params.main.long_query.spec) == {"obj", "floor"}:
        system_prompt = "You are a query parser named Digua. Your task is to parse a sentence into floor and object."
        prompt = f"Please parse the following sentence: {instruction}"
        prompt += "Output format: a comma-separated list in the order of floor and object. Example: [Floor 1, sofa]"
    elif set(params.main.long_query.spec) == {"obj"}:
        # return directly and not use the LLM for parsing
        print("floor, room, object:", None, None, instruction)
        return [None, None, instruction.strip()]

    conversation = Conversation(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    response = send_query(client, messages=conversation.messages, model=gpt_model, temperature=0.0)
    raw_result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")
    print("raw_result:", raw_result)
    # Split safely
    spec = set(params.main.long_query.spec)
    parts = [x.strip() for x in raw_result.split(",")]
    # Ensure always 3 elements
    floor, room, obj = None, None, None
    try:
        if spec == {"obj", "room", "floor"}:
            floor, room, obj = (parts + [None] * 3)[:3]
        elif spec == {"obj", "room"}:
            room, obj = (parts + [None] * 2)[:2]
        elif spec == {"obj", "floor"}:
            floor, obj = (parts + [None] * 2)[:2]
    except Exception as e:
        print(f"Warning: failed to parse LLM result: {raw_result}, error: {e}")
        floor, room, obj = None, None, instruction.strip()

    print("floor, room, object:", floor, room, obj)
    return (floor, room, obj)


def parse_hier_query_use_prompt_insentence_parse_icra(
    params, instruction: str
) -> Tuple[str, str, str]:
    """Parse long language query into a list of short queries at floor, room, and object level

    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")"""
    client, gpt_model = create_llm_client()

    # Depending on the query spec, parse the query differently:
    if set(params.main.long_query.spec) == {"obj", "room", "floor"}:
        system_prompt = "You are a query parser. Your task is to parse a sentence into floor, room, and object. If only room or object can be parsed, leave the other field empty. All descriptions except object must be in English."
        prompt = f"Please parse the following sentence: {instruction}"
        prompt += "Output format requirement: a list separated by commas, in the order of floor, room, and object. For example: [Floor 1, Horizon Exhibition Hall, sofa]"
    elif set(params.main.long_query.spec) == {"obj", "room"}:
        system_prompt = "You are a query parser named Diguo. Your task is to parse a sentence into room and object."
        prompt = f"Please parse the following sentence: {instruction}"
        prompt += "Output format requirement: a list separated by commas, in the order of room and object. For example: [Living Room, Sofa]"
    elif set(params.main.long_query.spec) == {"obj", "floor"}:
        system_prompt = "You are a query parser named Diguo. Your task is to parse a sentence into floor and object."
        prompt = f"Please parse the following sentence: {instruction}"
        prompt += "Output format requirement: a list separated by commas, in the order of floor and object. For example: [Floor 1, Sofa]"
    elif set(params.main.long_query.spec) == {"obj"}:
        # return directly and not use the LLM for parsing
        print("floor, room, object:", None, None, instruction)
        return [None, None, instruction.strip()]

    conversation = Conversation(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )

    response = send_query(client, messages=conversation.messages, model=gpt_model, temperature=0.0)
    raw_result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")
    print("raw_result:", raw_result)
    # Split safely
    spec = set(params.main.long_query.spec)
    parts = [x.strip() for x in raw_result.split(",")]
    # Ensure always 3 elements
    floor, room, obj = None, None, None
    try:
        if spec == {"obj", "room", "floor"}:
            floor, room, obj = (parts + [None] * 3)[:3]
        elif spec == {"obj", "room"}:
            room, obj = (parts + [None] * 2)[:2]
        elif spec == {"obj", "floor"}:
            floor, obj = (parts + [None] * 2)[:2]
    except Exception as e:
        print(f"Warning: failed to parse LLM result: {raw_result}, error: {e}")
        floor, room, obj = None, None, instruction.strip()

    print("floor, room, object:", floor, room, obj)
    return (floor, room, obj)


def parse_floor_room_object_gpt35(instruction: str) -> Tuple[str, str, str]:
    """Parse long language query into a list of short queries at floor, room, and object level

    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")"""

    client, gpt_model = create_llm_client()
    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "system",
                "content": "You are a hierarchical concept parser. You need to parse a description of an object into floor, region and object.",
            },
            {
                "role": "user",
                "content": "chair in region living room on the 0 floor",
            },
            {"role": "assistant", "content": "[floor 0,living room,chair]"},
            {
                "role": "user",
                "content": "floor in living room on floor 0",
            },
            {"role": "assistant", "content": "[floor 0,living room,floor]"},
            {
                "role": "user",
                "content": "table in kitchen on floor 3",
            },
            {"role": "assistant", "content": "[floor 3,kitchen,table]"},
            {
                "role": "user",
                "content": "cabinet in region bedroom on floor 1",
            },
            {"role": "assistant", "content": "[floor 1,bedroom,cabinet]"},
            {
                "role": "user",
                "content": "bedroom on floor 1",
            },
            {"role": "assistant", "content": "[floor 1,bedroom,]"},
            {
                "role": "user",
                "content": "bed",
            },
            {"role": "assistant", "content": "[,,bed]"},
            {
                "role": "user",
                "content": "bedroom",
            },
            {"role": "assistant", "content": "[,bedroom,]"},
            {
                "role": "user",
                "content": "I want to go to bed, where should I go?",
            },
            {"role": "assistant", "content": "[,bedroom,]"},
            {
                "role": "user",
                "content": "I want to go for something to eat upstairs. I am currently at floor 0, where should I go?",
            },
            {"role": "assistant", "content": "[floor 1,dinning,]"},
            {
                "role": "user",
                "content": f"{instruction}",
            },
        ],
    )
    print(response.choices[0].message.content)
    result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")
    print("floor, room, object:", result)
    decomposition = [x.strip() for x in result.split(",")]
    assert len(decomposition) == 3 and (
        decomposition[0] != "" or decomposition[1] != "" or decomposition[2] != ""
    )
    return decomposition


def parse_floor_room_object_gpt40(instruction: str) -> Tuple[str, str, str]:
    """Parse long language query into a list of short queries at floor, room, and object level

    Example: "mirror in region bathroom on floor 0" -> ("floor 0", "bathroom", "mirror")"""

    client, gpt_model = create_llm_client()

    response = client.chat.completions.create(
        model=gpt_model,
        messages=[
            {
                "role": "system",
                "content": "You are a hierarchical concept parser. You need to parse a description of an object into floor, region and object.",
            },
            {
                "role": "user",
                "content": "chair in region living room on the 0 floor",
            },
            {"role": "assistant", "content": "[floor 0,living room,chair]"},
            {
                "role": "user",
                "content": "floor in living room on floor 0",
            },
            {"role": "assistant", "content": "[floor 0,living room,floor]"},
            {
                "role": "user",
                "content": "table in kitchen on floor 3",
            },
            {"role": "assistant", "content": "[floor 3,kitchen,table]"},
            {
                "role": "user",
                "content": "cabinet in region bedroom on floor 1",
            },
            {"role": "assistant", "content": "[floor 1,bedroom,cabinet]"},
            {
                "role": "user",
                "content": "bedroom on floor 1",
            },
            {"role": "assistant", "content": "[floor 1,bedroom,]"},
            {
                "role": "user",
                "content": "bed",
            },
            {"role": "assistant", "content": "[,,bed]"},
            {
                "role": "user",
                "content": "bedroom",
            },
            {"role": "assistant", "content": "[,bedroom,]"},
            {
                "role": "user",
                "content": "I want to go to bed, where should I go?",
            },
            {"role": "assistant", "content": "[,bedroom,]"},
            {
                "role": "user",
                "content": "I want to go for something to eat upstairs. I am currently at floor 0, where should I go?",
            },
            {"role": "assistant", "content": "[floor 1,dinning,]"},
            {
                "role": "user",
                "content": f"{instruction}",
            },
        ],
    )
    print(response.choices[0].message.content)
    result = response.choices[0].message.content.strip().rstrip("]").lstrip("[")
    print("floor, room, object:", result)
    decomposition = [x.strip() for x in result.split(",")]
    assert len(decomposition) == 3 and (
        decomposition[0] != "" or decomposition[1] != "" or decomposition[2] != ""
    )
    return decomposition


def main():
    # result = parse_floor_room_object_gpt35("picture in region bedroom on floor 1")
    # object_list = ["sink", "soap", "towel", "hair dryer"]
    object_list = [
        "carpet",
        "counter",
        "baseball bat",
        "metal",
        "carpet",
        "banner",
        "blanket",
        "curtain",
        "dining table",
        "shelf",
        "cupboard",
        "curtain",
        "road",
        "banner",
        "banner",
        "oven",
        "carpet",
        "metal",
        "skateboard",
        "mirror",
        "bowl",
        "shelf",
        "mud",
        "cupboard",
        "window",
        "cupboard",
        "paper",
        "banner",
        "waterdrops",
        "waterdrops",
        "umbrella",
        "curtain",
        "refrigerator",
        "banner",
        "solid-other",
        "waterdrops",
        "clothes",
        "solid-other",
        "wood",
        "paper",
        "solid-other",
        "solid-other",
        "metal",
        "solid-other",
        "waterdrops",
        "bottle",
        "orange",
        "hat",
        "banner",
        "couch",
        "wood",
        "wood",
        "metal",
        "paper",
        "wood",
        "orange",
        "banner",
        "tv",
        "tv",
        "cupboard",
        "banner",
        "oven",
        "furniture-other",
        "cardboard",
        "metal",
        "banner",
        "hat",
        "curtain",
        "orange",
        "stone",
        "fog",
        "sink",
        "metal",
        "hat",
        "metal",
        "metal",
        "leaves",
    ]
    # default_list = ["guest room", "kitchen", "bathroom", "bedroom"]
    # result = infer_room_type_from_object_list(object_list, default_list)
    # result = infer_room_type_from_object_list_chat(object_list)
    # result = infer_floor_id_from_query([0, 1, 2, 3, 4], "floor 0")
    while True:
        instruction = input("Enter instruction: ")
        # result = parse_floor_room_object_gpt35(instruction)
        result = parse_floor_room_object_gpt40(instruction)
        print(result)
    # result = parse_floor_room_object_gpt35(
    #     "I want to cook something downstairs. I am currently at floor 1, where should I go?"
    # )
    # result = parse_floor_room_object_gpt35("cabinet")
    # print(result)


if __name__ == "__main__":
    main()
