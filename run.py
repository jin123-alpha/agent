import inspect
import json
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path

from tools import TOOL_MAP, TOOLS
from tools.memory_tools import format_memory_for_prompt


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:8b",

    # stream 只用于最终答案，不用于工具调用 JSON 判断
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
    memory_context = format_memory_for_prompt()

    return f"""
你是一个简单 Agent。

当前长期记忆：
{memory_context}

你可以使用工具：
{tool_descriptions}

记忆使用规则：
- 当用户明确要求你记住某个偏好、事实、项目约定或长期指令时，调用 remember。
- 当用户询问以前记住了什么，或当前问题可能依赖长期记忆时，调用 recall_memory。
- 当用户要求忘记某条记忆时，先查找对应记忆，再调用 forget_memory。
- 不要把密码、API key、访问令牌等敏感秘密写入长期记忆。

如果需要调用工具，只输出 JSON，不要输出其他文字：
{{"tool": "工具名", "arguments": {{"参数名": "参数值"}}}}

你可以连续调用多个工具。每次只调用一个工具，拿到工具结果后再判断是否继续调用工具。
当你已经获得足够信息时，直接给出最终回答，不要再输出 JSON。

当前可用工具名：{tool_names}

如果不需要工具，直接回答。
"""


def ask_model(messages):
    request_options = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }

    response = get_client().chat.completions.create(**request_options)
    return response.choices[0].message.content or ""


def stream_final_answer(final_answer: str) -> str:
    """
    如果 STREAM=False：直接返回最终答案，由主程序 print。
    如果 STREAM=True：用打字机效果输出最终答案，不再让模型复述一遍。
    """
    if not STREAM:
        return final_answer

    for char in final_answer:
        print(char, end="", flush=True)
        time.sleep(0.01)

    print()
    return ""


def clear_current_line():
    sys.stdout.write("\r" + " " * 80 + "\r")
    sys.stdout.flush()


@contextmanager
def tool_loading_line(tool_name: str):
    """
    工具调用时，只显示一行动态提示，不显示工具参数和 JSON。
    """
    stop_event = threading.Event()

    def animate():
        frames = ["", ".", "..", "..."]
        index = 0

        while not stop_event.is_set():
            frame = frames[index % len(frames)]
            sys.stdout.write(f"\r正在调用工具 {tool_name}{frame}")
            sys.stdout.flush()
            index += 1
            time.sleep(0.35)

    thread = threading.Thread(target=animate, daemon=True)
    thread.start()

    try:
        yield
    finally:
        stop_event.set()
        thread.join(timeout=1)
        clear_current_line()


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
        # 关键点：
        # 这里不能 stream=True。
        # 因为这一轮模型可能输出工具 JSON，如果流式输出，会把 JSON 暴露给用户。
        answer = ask_model(messages)

        action = parse_tool_action(answer)

        # 没有工具调用，说明这是最终答案
        if action is None:
            return stream_final_answer(answer)

        tool_name = action["tool"]
        arguments = action["arguments"]

        tool = TOOL_MAP.get(tool_name)
        if tool is None:
            return f"模型请求了未知工具：{tool_name}"

        # 工具执行时，只显示一行动态提示
        with tool_loading_line(tool_name):
            try:
                tool_result = tool(**arguments)
            except Exception as exc:
                tool_result = f"工具执行失败：{exc}"

        # 不把工具细节打印给用户，只加入上下文给模型
        messages.append({
            "role": "assistant",
            "content": answer,
        })
        messages.append({
            "role": "user",
            "content": (
                f"工具 {tool_name} 执行结果是：{tool_result}。\n"
                f"请判断是否还需要继续调用工具；如果不需要，请给出最终回答。"
            ),
        })

    messages.append({
        "role": "user",
        "content": "工具调用次数已达到上限。请基于已有信息给出最终回答。",
    })

    # 达到上限后的最终回答可以流式输出
    final_answer = ask_model(messages)
    return stream_final_answer(final_answer)


if __name__ == "__main__":
    result = run_agent("解读当前目录的文件结构并记下")

    if result:
        print(result)
