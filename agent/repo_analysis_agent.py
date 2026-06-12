"""RepoAnalysisAgent —— 仓库深度分析专家。"""

from handoff import Handoff
from session import Session
from tools import ToolRegistry

from .agent import Agent, create_agent_state_schema

AGENT_NAME = "RepoAnalysisAgent"

INSTRUCTIONS = """\
你是 RepoAnalysisAgent —— 仓库深度分析专家。

## 输入
你会收到一组仓库列表 JSON（由上一个 Agent 生成）。

## 任务
对每个仓库进行深度分析，提取以下维度的信息：
1. README 摘要（核心功能、特性亮点）
2. 项目结构（src/tests/docs/examples 是否齐全）
3. 依赖管理（requirements.txt / go.mod / package.json 等）
4. 部署支持（Docker / CI 配置）
5. 文档质量（是否有独立文档站点、API docs）
6. 许可证类型

## 输出格式（严格 JSON 数组）
[
  {
    "full_name": "owner/repo",
    "readme_summary": "项目核心功能描述...",
    "features": ["特性1", "特性2"],
    "has_tests": true,
    "has_docs": true,
    "has_ci": true,
    "has_docker": false,
    "license": "MIT",
    "dependency_files": ["requirements.txt"],
    "project_structure_quality": "良好/一般/较差"
  }
]

## 规则
- 使用 fetch_readme、list_repo_files 工具获取信息（如可用），否则使用 web_search。
- 客观分析，不要主观推荐。
- 只返回 JSON 数组，不要多余解释。
- 分析完成后，将结果交给下一个 Agent。
"""


def create_repo_analysis_graph(**kwargs):
    """
    RepoAnalysisAgent 的独立 LangGraph 流程。

    当前流程：
    START -> agent
    agent -- 有 tool_calls 且未超过 max_tool_calls --> tools -> agent
    agent -- 无 tool_calls 或达到上限 --> END

    以后可以在这里拆出 fetch_readme、list_repo_files、feature_extract
    等专用节点，让仓库分析拥有自己的流程。
    """
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
    AgentState = create_agent_state_schema(
        deps["Annotated"],
        deps["TypedDict"],
        deps["add_messages"],
    )

    langchain_tools = agent.tools.as_langchain_tools(deps["StructuredTool"])
    langchain_tool_map = {tool.name: tool for tool in langchain_tools}
    model_with_tools = build_model(ChatOpenAI, langchain_tools, config)
    tracer = agent.tracer or Tracer()

    def call_model(state):
        tracer.record("repo_analysis_model_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        tracer.record(
            "repo_analysis_model_end",
            has_tool_calls=bool(response.tool_calls),
        )
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def call_tools(state):
        last_message = state["messages"][-1]
        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = langchain_tool_map.get(tool_name)
            tracer.record("repo_analysis_tool_start", tool_name=tool_name)

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

            tracer.record("repo_analysis_tool_end", tool_name=tool_name)
            tool_messages.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_messages}

    def should_continue(state):
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


def create_repo_analysis_agent(
    session: Session | None = None,
    next_agent: str = "ScoringAgent",
) -> Agent:
    """创建仓库分析 Agent。"""
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
        graph_factory=create_repo_analysis_graph,
    )
