from __future__ import annotations

"""
CriticAgent —— 阶段性质量审查专家（语义层）。

纯规则校验（JSON 格式、字段完整性、范围检查）已下沉到 OutputGuardrail，
CriticAgent 只负责需要 LLM 语义理解的深层审查：
- 证据充分性检查
- 事实幻觉检测
- 数据来源可信度
- 跨 Agent 一致性
- 主观判断合理性
"""

from dataclasses import dataclass, field
from typing import Any

from session import Session
from tools import ToolRegistry, web_search

from .agent import Agent, create_agent_state_schema

AGENT_NAME = "CriticAgent"


# ---------------------------------------------------------------------------
# Critic 检查项（仅 LLM 语义层，规则层已在 OutputGuardrail）
# ---------------------------------------------------------------------------


@dataclass
class ChecklistItem:
    """单条检查项。"""

    name: str
    description: str
    required: bool = True


@dataclass
class StageChecklist:
    """某个阶段的语义审查清单。"""

    stage: str
    items: list[ChecklistItem] = field(default_factory=list)
    max_retries: int = 2


# --- 预定义检查清单（仅保留需要 LLM 判断的项） ---

SEARCH_CHECKLIST = StageChecklist(
    stage="GitHubSearchAgent",
    max_retries=2,
    items=[
        ChecklistItem(
            name="真实来源",
            description=(
                "每个仓库的 stars, language, updated_at 等事实字段必须来自真实工具或 GitHub API。"
                "如果缺少 source/api_url/retrieved_at 等来源字段，则需要通过查找对比来验证。"
            ),
        ),
        ChecklistItem(
            name="占位数据检查",
            description=(
                "检查 stars 和 updated_at 是否像占位数据或模型编造数据，"
                "例如 12345、9876、2025-01-01T00:00:00Z 等过于整齐的值。"
                "发现疑似占位数据时必须标记为 high severity。"
            ),
        ),
        ChecklistItem(
            name="硬约束满足",
            description=(
                "每个仓库必须满足用户硬约束（如 stars、language、更新时间等）。"
                "任何不满足硬约束的仓库必须标记为 failed。"
            ),
        ),
        ChecklistItem(
            name="主题相关性",
            description=(
                "仓库必须确实与用户需求相关。"
                "教程类仓库、资料合集、技能集合不能和框架类项目同等对待，应标记为相关性风险。"
            ),
            required=False,
        ),
    ],
)

ANALYSIS_CHECKLIST = StageChecklist(
    stage="RepoAnalysisAgent",
    max_retries=2,
    items=[
        ChecklistItem(
            name="证据字段",
            description=(
                "每个关键判断必须提供 evidence。包括 has_tests, has_docs, has_ci, "
                "has_docker, license, dependency_files, features。"
                "例如 has_docker=true 必须有 Dockerfile 或 docker-compose.yml 证据。"
            ),
        ),
        ChecklistItem(
            name="工程判断证据",
            description=(
                "has_tests 必须来自 tests/、test/ 或 CI 配置；"
                "has_docs 必须来自 docs/、README 链接或文档目录；"
                "has_ci 必须来自 .github/workflows、.gitlab-ci.yml 等；"
                "has_docker 必须来自 Dockerfile 或 docker-compose.yml。"
            ),
        ),
        ChecklistItem(
            name="特征具体性",
            description=(
                "features 不能只写 Lightweight architecture、Production ready 这类泛化描述，"
                "应尽量提取 supports_tools、supports_multi_agent、supports_memory 等可比较特征。"
            ),
            required=False,
        ),
        ChecklistItem(
            name="主观字段限制",
            description=(
                "project_structure_quality 这类主观字段必须有判断依据；"
                "如果没有量化规则或文件证据，应标记为不可靠。"
            ),
            required=False,
        ),
    ],
)

SCORING_CHECKLIST = StageChecklist(
    stage="ScoringAgent",
    max_retries=1,
    items=[
        ChecklistItem(
            name="权重可复核",
            description=(
                "必须提供评分权重或说明 weighted_total 的计算方式。"
                "如果没有 weights 字段或计算说明，应标记为证据不足。"
            ),
        ),
        ChecklistItem(
            name="评分依据",
            description=(
                "每个维度分数都应能追溯到 RepoAnalysisAgent 的字段或 evidence。"
                "不允许只给分数而不给理由。"
            ),
        ),
        ChecklistItem(
            name="硬约束降权",
            description=(
                "如果某仓库不满足用户硬约束，或只是教程/资料合集而非框架，"
                "requirement_match 不应获得高分。"
            ),
        ),
    ],
)

REPORT_CHECKLIST = StageChecklist(
    stage="ReportAgent",
    max_retries=1,
    items=[
        ChecklistItem(
            name="事实不新增",
            description=(
                "报告中的事实必须来自上游 Agent 数据。"
                "不得新增上游未提供的事实，例如官方维护、生产可用、更新频率最高等。"
            ),
        ),
        ChecklistItem(
            name="证据引用",
            description=(
                "推荐结论中应引用具体评分数据和证据来源。"
                "如果没有 README、GitHub metadata 或文件结构证据，应明确写为未确认。"
            ),
        ),
        ChecklistItem(
            name="推荐与排名一致",
            description=(
                "推荐结论中的排序必须与 ScoringAgent 的 rank 一致，"
                "不能擅自调整排名。"
            ),
        ),
        ChecklistItem(
            name="不确定性说明",
            description=(
                "报告必须包含未确认信息或风险说明，例如："
                "未实际运行项目、未统计 commit 频率、Docker 支持仅根据文件判断等。"
            ),
            required=False,
        ),
    ],
)

# stage → checklist 映射
STAGE_CHECKLISTS: dict[str, StageChecklist] = {
    "GitHubSearchAgent": SEARCH_CHECKLIST,
    "RepoAnalysisAgent": ANALYSIS_CHECKLIST,
    "ScoringAgent": SCORING_CHECKLIST,
    "ReportAgent": REPORT_CHECKLIST,
}


# ---------------------------------------------------------------------------
# Critic 指令模板
# ---------------------------------------------------------------------------


def build_critic_instructions(checklist: StageChecklist) -> str:
    """根据检查清单动态生成 Critic 指令。"""
    items_text = ""
    for i, item in enumerate(checklist.items, 1):
        required_tag = "【必须】" if item.required else "【建议】"
        items_text += f"{i}. {required_tag} {item.name}：{item.description}\n"

    return f"""\
你是 CriticAgent —— 语义质量审查专家，当前审查阶段：{checklist.stage}。

## 前置说明
输出的格式校验（JSON 合法性、字段完整性、数值范围）已由 OutputGuardrail 完成。
你无需重复检查格式问题，专注于以下语义层审查。

## 核心原则
你的目标不是表扬上游 Agent，而是尽可能发现：
- 事实错误或幻觉内容
- 证据不足（数据无来源支撑）
- 逻辑跳跃
- 过度确定的表述
- 不满足用户硬约束的结果
- 跨 Agent 数据不一致

## 审查要求
1. 没有来源支撑的事实字段必须写入 evidence_issues。
2. 如果无法确认事实，不要默认通过，应标记 warning 或 failed。
3. 对于【必须】项，只要存在严重证据不足，也应视为未通过。

## 检查清单
{items_text}

## 输出格式（严格 JSON）
{{
  "stage": "{checklist.stage}",
  "passed": true/false,
  "checks": [
    {{
      "name": "检查项名称",
      "passed": true/false,
      "required": true/false,
      "detail": "具体说明",
      "severity": "high/medium/low"
    }}
  ],
  "evidence_issues": ["证据问题1", "证据问题2"],
  "overall_comment": "总体评价",
  "retry_suggestion": "如果 passed=false，给出具体修改建议；如果 passed=true，设为 null"
}}

## 规则
- 所有【必须】项未通过 → passed=false
- 仅【建议】项未通过 → passed=true，但在 overall_comment 中指出
- 只输出 JSON，不要多余解释
"""


# ---------------------------------------------------------------------------
# Agent 工厂
# ---------------------------------------------------------------------------


def create_critic_agent(
    stage: str,
    session: Session | None = None,
) -> Agent:
    """
    创建针对特定阶段的 CriticAgent。

    Parameters
    ----------
    stage : str
        被审查的 Agent 名，如 "GitHubSearchAgent"
    """
    checklist = STAGE_CHECKLISTS.get(stage)
    if checklist is None:
        raise ValueError(
            f"No checklist defined for stage '{stage}'. "
            f"Available: {list(STAGE_CHECKLISTS.keys())}"
        )

    return Agent(
        name=f"{AGENT_NAME}_{stage}",
        instructions=build_critic_instructions(checklist),
        tools=ToolRegistry(tools=[web_search]),
        handoffs=[],
        session=session or Session(),
        graph_factory=create_critic_graph,
        metadata={"critic_stage": stage, "max_retries": checklist.max_retries},
    )


def create_critic_graph(**kwargs):
    """
    CriticAgent 的独立 LangGraph 流程。

    当前流程：
    START -> agent
    agent -- 有 tool_calls 且未超过 max_tool_calls --> tools -> agent
    agent -- 无 tool_calls 或达到上限 --> END

    Critic 和业务 agent 分开定义，是为了以后可以把它改成
    「审查 -> 必要时搜索核查 -> 复判 -> 输出审查 JSON」这样的专属流程。
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
        tracer.record("critic_model_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        tracer.record("critic_model_end", has_tool_calls=bool(response.tool_calls))
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
            tracer.record("critic_tool_start", tool_name=tool_name)

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

            tracer.record("critic_tool_end", tool_name=tool_name)
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


def get_checklist(stage: str) -> StageChecklist:
    """获取指定阶段的检查清单。"""
    return STAGE_CHECKLISTS[stage]
