from __future__ import annotations

"""ReportAgent —— 技术选型报告生成专家。"""

from session import Session
from tools import ToolRegistry

from .agent import Agent, create_agent_state_schema

AGENT_NAME = "ReportAgent"

INSTRUCTIONS = """\
你是 ReportAgent，负责把上游结构化数据整理为 Markdown 技术选型报告。

必须严格使用以下结构：
# 技术选型报告
## 1. 用户需求分析
## 2. 候选项目概览
## 3. GitHub 基础信息对比
## 4. 功能特性对比
## 5. 项目评分矩阵
## 6. 推荐方案
## 7. 风险分析
## 8. 后续开发路线建议
## 9. 参考来源

规则：
- 候选项目概览和评分矩阵必须使用 Markdown 表格。
- 评分使用 ScoringAgent 的 100 分制结果，不得自行改分或改排名。
- 所有事实必须来自上游 metadata、README/仓库证据或评分结果。
- 缺少 README、license 或其他信息时必须明确写“未确认”，不得补造。
- 参考来源必须列出 GitHub URL、README.md 或具体证据来源。
"""


def create_report_graph(**kwargs):
    """
    ReportAgent 的独立 LangGraph 流程。

    当前流程：
    START -> agent -> END

    以后如果报告生成要加入「生成草稿 -> 补证据 -> 输出 Markdown」
    等步骤，直接在本函数扩展。
    """
    from runner.langgraph_dependencies import import_langgraph_dependencies
    from runner.runner import build_chat_model
    from tracing import Tracer

    config = kwargs["config"]
    agent = kwargs["agent"]
    deps = import_langgraph_dependencies()
    HumanMessage = deps["HumanMessage"]
    SystemMessage = deps["SystemMessage"]
    ChatOpenAI = deps["ChatOpenAI"]
    END = deps["END"]
    START = deps["START"]
    StateGraph = deps["StateGraph"]
    AgentState = create_agent_state_schema(
        deps["Annotated"],
        deps["TypedDict"],
        deps["add_messages"],
    )

    model = build_chat_model(ChatOpenAI, config)
    tracer = agent.tracer or Tracer()

    def call_model(state):
        tracer.record("report_model_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model.invoke(messages)
        tracer.record("report_model_end")
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    return graph_builder.compile(), HumanMessage


def create_report_agent(session: Session | None = None) -> Agent:
    """创建报告生成 Agent。"""
    from guardrail.output_guardrails import STAGE_OUTPUT_GUARDRAILS

    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=ToolRegistry(tools=[]),
        handoffs=[],  # 流水线最后一个，无 handoff
        output_guardrails=STAGE_OUTPUT_GUARDRAILS.get(AGENT_NAME, []),
        session=session or Session(),
        graph_factory=create_report_graph,
    )
