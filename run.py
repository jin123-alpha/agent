import inspect
import json
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path

from tools import TOOL_MAP, TOOL_SPECS, TOOLS


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:8b",

    # stream 只用于最终答案，不用于工具调用 JSON 判断
    "stream": False,
    "native_tools": True,
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
    loaded_config["native_tools"] = parse_bool(loaded_config["native_tools"])
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
NATIVE_TOOLS = CONFIG["native_tools"]

client = None
native_tools_available = NATIVE_TOOLS


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


def build_system_prompt(use_native_tools: bool = NATIVE_TOOLS) -> str:
    if use_native_tools:
        tool_names = ", ".join(tool.__name__ for tool in TOOLS)
        return f"""
你是一个简单 Agent。

记忆使用规则：
- 当用户明确要求你记住某个偏好、事实、项目约定或长期指令时，调用 remember。
- 当用户询问以前记住了什么，或当前问题可能依赖长期记忆时，调用 recall_memory。
- 当用户要求忘记某条记忆时，先查找对应记忆，再调用 forget_memory。
- 不要把密码、API key、访问令牌等敏感秘密写入长期记忆。

你可以使用原生工具调用。需要外部信息、读写文件、查询记忆或保存记忆时，选择最合适的工具。
不要在最终回答中直接输出大段代码；需要创建或修改文件时使用文件工具。

当前可用工具名：{tool_names}

如果不需要工具，直接回答。
"""

    tool_descriptions = "\n".join(format_tool_description(tool) for tool in TOOLS)
    tool_names = ", ".join(tool.__name__ for tool in TOOLS)

    return f"""
你是一个简单 Agent。

你可以使用工具：
{tool_descriptions}

记忆使用规则：
- 当用户明确要求你记住某个偏好、事实、项目约定或长期指令时，调用 remember。
- 当用户询问以前记住了什么，或当前问题可能依赖长期记忆时，调用 recall_memory。
- 当用户要求忘记某条记忆时，先查找对应记忆，再调用 forget_memory。
- 不要把密码、API key、访问令牌等敏感秘密写入长期记忆。
- 不要默认把长期记忆放入上下文；只有需要时才调用 recall_memory。

如果需要调用工具，只输出 JSON，不要输出其他文字：
{{"tool": "工具名", "arguments": {{"参数名": "参数值"}}}}

你可以连续调用多个工具。每次只调用一个工具，拿到工具结果后再判断是否继续调用工具。
当你已经获得足够信息时，直接给出最终回答，不要再输出 JSON。

当前可用工具名：{tool_names}

如果不需要工具，直接回答。
"""


def ask_model(messages, use_native_tools: bool = False):
    request_options = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
    }
    if use_native_tools:
        request_options["tools"] = TOOL_SPECS
        request_options["tool_choice"] = "auto"

    response = get_client().chat.completions.create(**request_options)
    return response.choices[0].message


def get_value(source, name: str, default=None):
    if isinstance(source, dict):
        return source.get(name, default)

    return getattr(source, name, default)


def stream_model_response(messages, use_native_tools: bool = False) -> dict:
    request_options = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "stream": True,
    }
    if use_native_tools:
        request_options["tools"] = TOOL_SPECS
        request_options["tool_choice"] = "auto"

    response = get_client().chat.completions.create(**request_options)
    content_parts = []
    tool_call_parts = {}

    for chunk in response:
        choices = get_value(chunk, "choices", [])
        if not choices:
            continue

        delta = get_value(choices[0], "delta")
        if delta is None:
            continue

        content = get_value(delta, "content")
        if content:
            print(content, end="", flush=True)
            content_parts.append(content)

        for tool_call in get_value(delta, "tool_calls", []) or []:
            index = get_value(tool_call, "index", 0) or 0
            current = tool_call_parts.setdefault(index, {
                "id": "",
                "type": "function",
                "function": {
                    "name": "",
                    "arguments": "",
                },
            })

            call_id = get_value(tool_call, "id")
            if call_id:
                current["id"] = call_id

            call_type = get_value(tool_call, "type")
            if call_type:
                current["type"] = call_type

            function = get_value(tool_call, "function")
            if function is None:
                continue

            name = get_value(function, "name")
            if name:
                current["function"]["name"] += name

            arguments = get_value(function, "arguments")
            if arguments:
                current["function"]["arguments"] += arguments

    content = "".join(content_parts)
    if content and not tool_call_parts:
        print()

    message = {
        "role": "assistant",
        "content": content,
    }
    if tool_call_parts:
        message["tool_calls"] = [
            tool_call_parts[index]
            for index in sorted(tool_call_parts)
        ]

    return message


def message_to_dict(message) -> dict:
    if isinstance(message, dict):
        return {
            key: value
            for key, value in message.items()
            if value is not None
        }

    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)

    return {
        "role": getattr(message, "role", "assistant"),
        "content": getattr(message, "content", "") or "",
    }


def message_content(message) -> str:
    if isinstance(message, dict):
        return message.get("content") or ""

    return getattr(message, "content", None) or ""


def get_native_tool_calls(message) -> list:
    if isinstance(message, dict):
        return message.get("tool_calls") or []

    return getattr(message, "tool_calls", None) or []


def tool_call_id(tool_call) -> str:
    if isinstance(tool_call, dict):
        return tool_call.get("id", "")

    return getattr(tool_call, "id", "")


def tool_call_name_and_arguments(tool_call) -> tuple[str, dict]:
    if isinstance(tool_call, dict):
        function = tool_call.get("function") or {}
        name = function.get("name", "")
        raw_arguments = function.get("arguments") or "{}"
    else:
        function = getattr(tool_call, "function", None)
        name = getattr(function, "name", "") if function else ""
        raw_arguments = getattr(function, "arguments", "{}") if function else "{}"

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError:
        arguments = {}

    if not isinstance(arguments, dict):
        arguments = {}

    return name, arguments


def execute_tool(tool_name: str, arguments: dict) -> str:
    tool = TOOL_MAP.get(tool_name)
    if tool is None:
        return f"模型请求了未知工具：{tool_name}"

    with tool_loading_line(tool_name):
        try:
            return str(tool(**arguments))
        except Exception as exc:
            return f"工具执行失败：{exc}"


def stream_final_answer(final_answer: str) -> str:
    """
    如果 STREAM=False：直接返回最终答案，由主程序 print。
    如果 STREAM=True：输出已经缓冲好的最终答案；原生工具路径会优先使用 API 真流式。
    """
    if not STREAM:
        return final_answer

    if final_answer:
        print(final_answer)
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
        sys.stdout.write("\n\n")
        sys.stdout.flush()


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
    global native_tools_available

    messages = [
        {
            "role": "system",
            "content": build_system_prompt(native_tools_available),
        },
        {
            "role": "user",
            "content": user_input,
        },
    ]

    for _ in range(max_tool_calls):
        # JSON 兜底模式不能流式输出工具决策，否则会把工具 JSON 暴露给用户。
        # 原生 tools 模式可以流式输出普通文本，并在后台拼接 tool_calls。
        answer_already_streamed = False
        if native_tools_available:
            try:
                if STREAM:
                    answer_message = stream_model_response(messages, use_native_tools=True)
                    answer_already_streamed = True
                else:
                    answer_message = ask_model(messages, use_native_tools=True)
            except Exception:
                native_tools_available = False
                messages[0]["content"] = build_system_prompt(use_native_tools=False)
                answer_message = ask_model(messages)

            tool_calls = get_native_tool_calls(answer_message)
            if tool_calls:
                messages.append(message_to_dict(answer_message))

                for tool_call in tool_calls:
                    tool_name, arguments = tool_call_name_and_arguments(tool_call)
                    tool_result = execute_tool(tool_name, arguments)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id(tool_call),
                        "content": tool_result,
                    })

                continue
        else:
            answer_message = ask_model(messages)

        answer = message_content(answer_message)
        action = parse_tool_action(answer)

        # 没有工具调用，说明这是最终答案
        if action is None:
            if answer_already_streamed:
                return ""
            return stream_final_answer(answer)

        tool_name = action["tool"]
        arguments = action["arguments"]

        tool_result = execute_tool(tool_name, arguments)

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
    if STREAM and native_tools_available:
        stream_model_response(messages)
        return ""

    final_answer = message_content(ask_model(messages))
    return stream_final_answer(final_answer)


if __name__ == "__main__":
    result = run_agent("上海嘉定今天天气如何")

    if result:
        print(result)
