"""ReportAgent —— 技术选型报告生成专家。"""

from session import Session
from tools import ToolRegistry

from .agent import Agent, create_agent_state_schema

AGENT_NAME = "ReportAgent"

INSTRUCTIONS = """\
你是 ReportAgent —— 技术选型报告生成专家。

## 输入
你会收到完整的流水线数据：需求 JSON、仓库列表、分析结果、评分结果。

## 任务
生成一份专业的 Markdown 技术选型报告。

## 报告结构
```
# 开源技术选型报告

## 1. 需求概述
（用自然语言描述用户需求）

## 2. 候选项目概览
| 排名 | 项目 | Stars | 语言 | 许可证 | 加权总分 |
|------|------|-------|------|--------|---------|

## 3. 详细分析

### 3.1 [项目名]
- **核心功能**：...
- **优势**：...
- **不足**：...
- **评分明细**：需求匹配 X/10 | 社区 X/10 | 文档 X/10 | 工程 X/10 | 部署 X/10

（每个项目重复此结构）

## 4. 推荐结论
（给出 Top 1-2 推荐，说明理由）

## 5. 注意事项
（风险提示、许可证合规、维护状态等）
```

## 规则
- 报告必须结构完整，内容翔实。
- 所有数据必须有据可查（来自前序 Agent 的分析结果）。
- 推荐结论要给出明确理由。
- 输出完整 Markdown 报告。
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
