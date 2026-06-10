import inspect
import json

from tools import TOOL_MAP, TOOLS


BASE_URL = "http://10.6.22.1:11434/v1"
API_KEY = "ollama"
MODEL = "qwen3:8b"

client = None


def get_client():
    global client

    if client is None:
        from openai import OpenAI

        client = OpenAI(
            base_url=BASE_URL,
            api_key=API_KEY,
        )

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


def ask_model(messages):
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content


def parse_tool_action(answer: str):
    try:
        action = json.loads(answer)
    except json.JSONDecodeError:
        return None

    if not isinstance(action, dict):
        return None

    if "tool" not in action or "arguments" not in action:
        return None

    if not isinstance(action["arguments"], dict):
        return None

    return action


def run_agent(user_input: str, max_tool_calls: int = 8):
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
            return answer

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
    return ask_model(messages)


if __name__ == "__main__":
    print(run_agent("如何寄送相机等高价值物品？"))
