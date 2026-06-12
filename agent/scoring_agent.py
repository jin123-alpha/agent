"""ScoringAgent —— 项目评分排名专家。"""

from handoff import Handoff
from session import Session
from tools import ToolRegistry

from .agent import Agent

AGENT_NAME = "ScoringAgent"

INSTRUCTIONS = """\
你是 ScoringAgent —— 项目评分排名专家。

## 输入
你会收到需求 JSON 和仓库分析结果 JSON（由前序 Agent 生成）。

## 任务
根据以下维度对每个仓库打分（每项 0-10 分），并计算加权总分：

| 维度 | 权重 | 说明 |
|------|------|------|
| 需求匹配度 | 30% | features 与用户 keywords/constraints 的匹配程度 |
| 社区活跃度 | 20% | stars、更新时间 |
| 文档质量   | 20% | has_docs、readme_summary 质量 |
| 工程规范   | 15% | has_tests、has_ci、project_structure_quality |
| 部署友好度 | 15% | has_docker、dependency_files 规范程度 |

## 输出格式（严格 JSON 数组，按总分降序）
[
  {
    "full_name": "owner/repo",
    "scores": {
      "requirement_match": 8,
      "community_activity": 9,
      "doc_quality": 7,
      "engineering": 8,
      "deployment": 6
    },
    "weighted_total": 7.85,
    "rank": 1
  }
]

## 规则
- weighted_total = 各项分数 × 对应权重之和，保留两位小数。
- 必须按 weighted_total 降序排列并标注 rank。
- 只返回 JSON 数组，不要多余解释。
- 评分完成后，将结果交给下一个 Agent。
"""


def create_scoring_graph(**kwargs):
    """
    ScoringAgent 的独立 LangGraph 流程。

    当前流程：
    START -> agent -> END

    以后如果评分要改成「规则打基础分 -> 模型解释 -> guardrail 校验」，
    直接在这里添加节点和边。
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
    add_messages = deps["add_messages"]
    Annotated = deps["Annotated"]
    TypedDict = deps["TypedDict"]

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        llm_calls: int

    model = build_chat_model(ChatOpenAI, config)
    tracer = agent.tracer or Tracer()

    def call_model(state: AgentState):
        tracer.record("scoring_model_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model.invoke(messages)
        tracer.record("scoring_model_end")
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    return graph_builder.compile(), HumanMessage


def create_scoring_agent(
    session: Session | None = None,
    next_agent: str = "ReportAgent",
) -> Agent:
    """创建评分 Agent。"""
    from guardrail.output_guardrails import STAGE_OUTPUT_GUARDRAILS
    from .agents import _make_handoff

    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=ToolRegistry(tools=[]),
        handoffs=[_make_handoff(AGENT_NAME, next_agent)],
        output_guardrails=STAGE_OUTPUT_GUARDRAILS.get(AGENT_NAME, []),
        session=session or Session(),
        graph_factory=create_scoring_graph,
    )
