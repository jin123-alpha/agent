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
