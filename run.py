import json
import sys
import time
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from tools import TOOLS


CONFIG_PATH = Path(__file__).resolve().parent / "config.json"
DEFAULT_CONFIG = {
    "base_url": "http://127.0.0.1:11434/v1",
    "api_key": "ollama",
    "model": "qwen3:8b",

    # LangGraph streams LLM message chunks when this is true.
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

_compiled_graphs = {}


def import_langgraph_dependencies():
    try:
        from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
        from langchain_core.tools import StructuredTool
        from langchain_openai import ChatOpenAI
        from langgraph.graph import END, START, StateGraph, add_messages
        from typing_extensions import Annotated, TypedDict
    except ImportError as exc:
        raise RuntimeError(
            "缺少 LangGraph/LangChain 依赖。请先安装："
            "pip install -r requirements.txt"
        ) from exc

    return {
        "HumanMessage": HumanMessage,
        "SystemMessage": SystemMessage,
        "ToolMessage": ToolMessage,
        "StructuredTool": StructuredTool,
        "ChatOpenAI": ChatOpenAI,
        "END": END,
        "START": START,
        "StateGraph": StateGraph,
        "add_messages": add_messages,
        "Annotated": Annotated,
        "TypedDict": TypedDict,
    }


def build_system_prompt() -> str:
    tool_names = ", ".join(tool.__name__ for tool in TOOLS)

    return f"""
你是一个简单 Agent。

记忆使用规则：
- 当用户明确要求你记住某个偏好、事实、项目约定或长期指令时，调用 remember。
- 当用户询问以前记住了什么，或当前问题可能依赖长期记忆时，调用 recall_memory。
- 当用户要求忘记某条记忆时，先查找对应记忆，再调用 forget_memory。
- 不要把密码、API key、访问令牌等敏感秘密写入长期记忆。
- 不要默认把长期记忆放入上下文；只有需要时才调用 recall_memory。

需要外部信息、读写文件、查询记忆或保存记忆时，选择最合适的工具。
不要在最终回答中直接输出大段代码；需要创建或修改文件时使用文件工具。

当前可用工具名：{tool_names}

如果不需要工具，直接回答。
"""


def build_langchain_tools(StructuredTool):
    return [
        StructuredTool.from_function(
            func=tool,
            name=tool.__name__,
            description=(tool.__doc__ or "").strip() or None,
        )
        for tool in TOOLS
    ]


def build_model(ChatOpenAI, langchain_tools):
    model_options = {
        "model": MODEL,
        "api_key": API_KEY,
        "temperature": 0,
    }
    if BASE_URL:
        model_options["base_url"] = BASE_URL

    return ChatOpenAI(**model_options).bind_tools(langchain_tools)


def create_agent_graph(max_tool_calls: int = 10):
    if max_tool_calls in _compiled_graphs:
        return _compiled_graphs[max_tool_calls]

    deps = import_langgraph_dependencies()
    HumanMessage = deps["HumanMessage"]
    SystemMessage = deps["SystemMessage"]
    ToolMessage = deps["ToolMessage"]
    StructuredTool = deps["StructuredTool"]
    ChatOpenAI = deps["ChatOpenAI"]
    END = deps["END"]
    START = deps["START"]
    StateGraph = deps["StateGraph"]
    add_messages = deps["add_messages"]
    Annotated = deps["Annotated"]
    TypedDict = deps["TypedDict"]

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        llm_calls: int

    langchain_tools = build_langchain_tools(StructuredTool)
    langchain_tool_map = {tool.name: tool for tool in langchain_tools}
    model_with_tools = build_model(ChatOpenAI, langchain_tools)

    def call_model(state: AgentState):
        messages = [
            SystemMessage(content=build_system_prompt()),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def call_tools(state: AgentState):
        last_message = state["messages"][-1]
        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = langchain_tool_map.get(tool_name)

            if tool is None:
                tool_result = f"模型请求了未知工具：{tool_name}"
            else:
                with tool_loading_line(tool_name):
                    try:
                        tool_result = str(tool.invoke(tool_call["args"]))
                    except Exception as exc:
                        tool_result = f"工具执行失败：{exc}"

            tool_messages.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls and state.get("llm_calls", 0) <= max_tool_calls:
            return "tools"

        return END

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", call_tools)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
    graph_builder.add_edge("tools", "agent")

    graph = graph_builder.compile()
    _compiled_graphs[max_tool_calls] = (graph, HumanMessage)
    return _compiled_graphs[max_tool_calls]


def final_message_content(messages) -> str:
    for message in reversed(messages):
        content = getattr(message, "content", "")
        if isinstance(content, str) and content.strip():
            return content

    return ""


def run_agent(user_input: str, max_tool_calls: int = 10):
    graph, HumanMessage = create_agent_graph(max_tool_calls)
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "llm_calls": 0,
    }

    if STREAM:
        printed = False
        for part in graph.stream(
            initial_state,
            stream_mode="messages",
            version="v2",
        ):
            if part["type"] != "messages":
                continue

            message_chunk, metadata = part["data"]
            if metadata.get("langgraph_node") != "agent":
                continue

            content = getattr(message_chunk, "content", "")
            if not content:
                continue

            print(content, end="", flush=True)
            printed = True

        if printed:
            print()
        return ""

    final_state = graph.invoke(initial_state)
    return final_message_content(final_state["messages"])


@contextmanager
def tool_loading_line(tool_name: str):
    """
    工具调用时，只显示一行动态提示，不显示工具参数。
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


if __name__ == "__main__":
    result = run_agent("上海嘉定今天天气如何")

    if result:
        print(result)
