"""GitHubSearchAgent —— GitHub 仓库搜索专家。"""

from handoff import Handoff
from session import Session
from tools import ToolRegistry

from .agent import Agent

AGENT_NAME = "GitHubSearchAgent"

INSTRUCTIONS = """\
你是 GitHubSearchAgent —— GitHub 仓库搜索专家。

## 输入
你会收到一段结构化需求 JSON（由上一个 Agent 生成）。

## 任务
根据需求中的 keywords、language、constraints，搜索 GitHub 仓库，返回 top_n 个最相关仓库。

## 输出格式（严格 JSON 数组）
[
  {
    "full_name": "owner/repo",
    "html_url": "https://github.com/owner/repo",
    "description": "仓库描述",
    "stars": 12345,
    "language": "Python",
    "updated_at": "2025-01-01T00:00:00Z"
  }
]

## 规则
- 使用 search_github_repos 工具搜索（如可用），否则使用 web_search。
- 优先按 stars 排序。
- 只返回 JSON 数组，不要多余解释。
- 搜索完成后，将结果交给下一个 Agent。
"""


def create_github_search_graph(**kwargs):
    """
    GitHubSearchAgent 的独立 LangGraph 流程。

    当前流程：
    START -> agent
    agent -- 有 tool_calls 且未超过 max_tool_calls --> tools -> agent
    agent -- 无 tool_calls 或达到上限 --> END

    以后如果 GitHub 搜索要改成「先构造查询 -> 调工具 -> 去重排序 -> 总结」，
    就在本函数里增加对应节点，不影响其他 agent。
    """
    from typing import Literal

    from runner.langgraph_dependencies import import_langgraph_dependencies
    from runner.runner import build_model
    from tracing import Tracer

    config = kwargs["config"]
    max_tool_calls = kwargs.get("max_tool_calls", 10)
    agent = kwargs["agent"]
    on_tool_start = kwargs.get("on_tool_start")
    on_tool_end = kwargs.get("on_tool_end")

    deps = import_langgraph_dependencies()
    HumanMessage = deps["HumanMessage"]
    SystemMessage = deps["SystemMessage"]
    ToolMessage = deps["ToolMessage"]
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

    langchain_tools = agent.tools.as_langchain_tools(deps["StructuredTool"])
    langchain_tool_map = {tool.name: tool for tool in langchain_tools}
    model_with_tools = build_model(ChatOpenAI, langchain_tools, config)
    tracer = agent.tracer or Tracer()

    def call_model(state: AgentState):
        tracer.record("github_search_model_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        tracer.record(
            "github_search_model_end",
            has_tool_calls=bool(response.tool_calls),
        )
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
            tracer.record("github_search_tool_start", tool_name=tool_name)

            if tool is None:
                tool_result = f"模型请求了未知工具：{tool_name}"
            else:
                if on_tool_start:
                    on_tool_start(tool_name)

                try:
                    tool_result = str(tool.invoke(tool_call["args"]))
                except Exception as exc:
                    tool_result = f"工具执行失败：{exc}"
                finally:
                    if on_tool_end:
                        on_tool_end(tool_name)

            tracer.record("github_search_tool_end", tool_name=tool_name)
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

    return graph_builder.compile(), HumanMessage


def create_github_search_agent(
    session: Session | None = None,
    next_agent: str = "RepoAnalysisAgent",
) -> Agent:
    """创建 GitHub 搜索 Agent。"""
    from tools import web_search

    from guardrail.output_guardrails import STAGE_OUTPUT_GUARDRAILS
    from .agents import _make_handoff

    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=ToolRegistry(tools=[web_search]),
        handoffs=[_make_handoff(AGENT_NAME, next_agent)],
        output_guardrails=STAGE_OUTPUT_GUARDRAILS.get(AGENT_NAME, []),
        session=session or Session(),
        graph_factory=create_github_search_graph,
    )
