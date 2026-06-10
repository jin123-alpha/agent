import inspect
import json
from pathlib import Path

from tools import TOOL_MAP, TOOLS


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CONFIG = {
    "base_url": "http://10.6.22.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:8b",
    "stream": False,
}


def load_config() -> dict:
    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config = json.load(file)
    except Exception:
        return DEFAULT_CONFIG.copy()

    if not isinstance(config, dict):
        return DEFAULT_CONFIG.copy()

    loaded_config = DEFAULT_CONFIG.copy()
    loaded_config.update({
        key: value
        for key, value in config.items()
        if key in loaded_config
    })

    loaded_config["stream"] = parse_bool(loaded_config["stream"])
    return loaded_config


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}

    return bool(value)


CONFIG = load_config()
BASE_URL = CONFIG["base_url"]
API_KEY = CONFIG["api_key"]
MODEL = CONFIG["model"]
STREAM = CONFIG["stream"]

client = None


def get_client():
    global client

    if client is None:
        from openai import OpenAI

        client_options = {"api_key": API_KEY}
        if BASE_URL:
            client_options["base_url"] = BASE_URL

        client = OpenAI(**client_options)

    return client


def format_tool_description(tool) -> str:
    signature = inspect.signature(tool)
    description = inspect.getdoc(tool) or "无描述。"
    return f"- {tool.__name__}{signature}：{description}"


def build_system_prompt() -> str:
    tool_descriptions = "\n".join(format_tool_description(tool) for tool in TOOLS)
    tool_names = ", ".join(tool.__name__ for tool in TOOLS)

    return f"""
你是一个简单 Agent。

你可以使用工具：
{tool_descriptions}

如果需要调用工具，只输出 JSON，不要输出其他文字：
{{"tool": "工具名", "arguments": {{"参数名": "参数值"}}}}

你可以连续调用多个工具。每次只调用一个工具，拿到工具结果后再判断是否继续调用工具。
当你已经获得足够信息时，直接给出最终回答，不要再输出 JSON。

当前可用工具名：{tool_names}

如果不需要工具，直接回答。
"""


def ask_model(messages, stream_output: bool = False):
    request_options = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }

    if not STREAM:
        response = get_client().chat.completions.create(**request_options)
        return response.choices[0].message.content

    chunks = get_client().chat.completions.create(
        **request_options,
        stream=True,
    )
    content_parts = []

    for chunk in chunks:
        delta = chunk.choices[0].delta.content
        if delta:
            if stream_output:
                print(delta, end="", flush=True)
            content_parts.append(delta)

    if stream_output:
        print()
    return "".join(content_parts)


def stream_final_answer(final_answer: str) -> str:
    if not STREAM:
        return final_answer

    messages = [
        {
            "role": "system",
            "content": "你只负责把给定的最终答案原样输出给用户，不要添加解释，不要输出 JSON。",
        },
        {
            "role": "user",
            "content": final_answer,
        },
    ]
    ask_model(messages, stream_output=True)
    return ""


def parse_tool_action(answer: str):
    action = parse_json_action(answer)

    if action is not None:
        return action

    decoder = json.JSONDecoder()

    for index, char in enumerate(answer):
        if char != "{":
            continue

        try:
            candidate, _ = decoder.raw_decode(answer[index:])
        except json.JSONDecodeError:
            continue

        action = parse_json_action(candidate)
        if action is not None:
            return action

    return None


def parse_json_action(answer):
    if isinstance(answer, dict):
        action = answer
    else:
        try:
            action = json.loads(answer)
        except (TypeError, json.JSONDecodeError):
            return None

    if not isinstance(action, dict):
        return None
    
    if "tool" not in action or "arguments" not in action:
        return None

    if not isinstance(action["arguments"], dict):
        return None

    return action


def run_agent(user_input: str, max_tool_calls: int = 10):
    messages = [
        {
            "role": "system",
            "content": build_system_prompt(),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    for _ in range(max_tool_calls):
        answer = ask_model(messages)
        action = parse_tool_action(answer)

        if action is None:
            return stream_final_answer(answer)

        tool_name = action["tool"]
        arguments = action["arguments"]

        tool = TOOL_MAP.get(tool_name)
        if tool is None:
            return f"模型请求了未知工具：{tool_name}"

        try:
            tool_result = tool(**arguments)
        except Exception as exc:
            tool_result = f"工具执行失败：{exc}"

        messages.append({
            "role": "assistant",
            "content": answer,
        })
        messages.append({
            "role": "user",
            "content": f"工具 {tool_name} 执行结果是：{tool_result}。请判断是否还需要继续调用工具；如果不需要，请给出最终回答。",
        })

    messages.append({
        "role": "user",
        "content": "工具调用次数已达到上限。请基于已有信息给出最终回答。",
    })
    final_answer = ask_model(messages, stream_output=STREAM)
    return "" if STREAM else final_answer


if __name__ == "__main__":
    result = run_agent("检查本地目录，告诉我写了一个什么项目")
    if result:
        print(result,end='',flush=STREAM)
