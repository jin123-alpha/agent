"""ScoringAgent: deterministic scoring and validation state machine."""

from __future__ import annotations

import json
import re
from typing import Any
from typing import Annotated
from session import Session
from tools import ToolRegistry, score_projects, validate_project_result

from .agent import Agent

AGENT_NAME = "ScoringAgent"

INSTRUCTIONS = """\
你是 ScoringAgent，负责使用固定、可复核的规则给候选项目评分。

评分维度与满分：
- 功能匹配度 function_match：30
- 部署便利性 deployment：20
- 二次开发友好度 developer_friendliness：20
- 社区活跃度 community_activity：15
- 文档完善度 documentation：10
- 许可证友好度 license_friendliness：5

你必须使用 score_projects 的确定性结果，不得自行修改分数。输出为 JSON 数组，
每项包含 project、total_score、scores、reason、evidence、weights 和 rank。
"""


def _decode_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _extract_context(prompt: str) -> dict[str, Any]:
    match = re.search(r"```json\s*(.*?)\s*```", prompt, re.DOTALL)
    if match:
        parsed = _decode_json(match.group(1))
        if isinstance(parsed, dict):
            return parsed

    parsed = _decode_json(prompt)
    return parsed if isinstance(parsed, dict) else {}


def create_scoring_graph(**kwargs):
    """
    Build the scoring state machine.

    START -> parse_input -> score_projects -> validate_scores -> finalize -> END
    """
    from runner.langgraph_dependencies import import_langgraph_dependencies
    from tracing import Tracer

    agent = kwargs["agent"]
    deps = import_langgraph_dependencies()
    AIMessage = deps["AIMessage"]
    HumanMessage = deps["HumanMessage"]
    END = deps["END"]
    START = deps["START"]
    StateGraph = deps["StateGraph"]
    Annotated = deps["Annotated"]
    TypedDict = deps["TypedDict"]
    add_messages = deps["add_messages"]
    tracer = agent.tracer or Tracer()

    class ScoringState(TypedDict, total=False):
        messages: Annotated[list, add_messages]
        llm_calls: int
        requirements: dict[str, Any]
        projects: list[dict[str, Any]]
        metadata_projects: list[dict[str, Any]]
        scores_json: str
        validation: dict[str, Any]

    def parse_input(state: ScoringState):
        prompt = str(state["messages"][-1].content)
        context = _extract_context(prompt)
        requirements = _decode_json(context.get("RequirementAgent", {}))
        projects = _decode_json(context.get("RepoAnalysisAgent", []))
        metadata_projects = _decode_json(context.get("GitHubSearchAgent", []))

        if not context:
            direct = _decode_json(prompt)
            if isinstance(direct, dict):
                requirements = _decode_json(direct.get("RequirementAgent", {}))
                projects = _decode_json(
                    direct.get("RepoAnalysisAgent", direct.get("projects", []))
                )
                metadata_projects = _decode_json(
                    direct.get("GitHubSearchAgent", direct.get("project_metadata", []))
                )

        tracer.record(
            "scoring_parse_input",
            project_count=len(projects) if isinstance(projects, list) else 0,
        )
        return {
            "requirements": requirements if isinstance(requirements, dict) else {},
            "projects": projects if isinstance(projects, list) else [],
            "metadata_projects": (
                metadata_projects if isinstance(metadata_projects, list) else []
            ),
        }

    def calculate_scores(state: ScoringState):
        tracer.record("scoring_rules_start")
        scores_json = score_projects(
            state.get("projects", []),
            state.get("requirements", {}),
            state.get("metadata_projects", []),
        )
        tracer.record("scoring_rules_end")
        return {"scores_json": scores_json}

    def validate_scores(state: ScoringState):
        validation = json.loads(
            validate_project_result(
                state.get("scores_json", "[]"),
                state.get("projects", []),
            )
        )
        tracer.record(
            "scoring_validation",
            ok=validation["ok"],
            error_count=len(validation["errors"]),
            warning_count=len(validation["warnings"]),
        )
        return {"validation": validation}

    def finalize(state: ScoringState):
        validation = state.get("validation", {})
        if not validation.get("ok", False):
            payload = {
                "error": "scoring_validation_failed",
                "details": validation.get("errors", []),
            }
            output = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            output = state.get("scores_json", "[]")
        return {"messages": [AIMessage(content=output)]}

    graph_builder = StateGraph(ScoringState)
    graph_builder.add_node("parse_input", parse_input)
    graph_builder.add_node("score_projects", calculate_scores)
    graph_builder.add_node("validate_scores", validate_scores)
    graph_builder.add_node("finalize", finalize)
    graph_builder.add_edge(START, "parse_input")
    graph_builder.add_edge("parse_input", "score_projects")
    graph_builder.add_edge("score_projects", "validate_scores")
    graph_builder.add_edge("validate_scores", "finalize")
    graph_builder.add_edge("finalize", END)
    return graph_builder.compile(), HumanMessage


def create_scoring_agent(
    session: Session | None = None,
    next_agent: str = "CriticAgent_ScoringAgent",
) -> Agent:
    """Create the deterministic scoring agent."""
    from guardrail.output_guardrails import STAGE_OUTPUT_GUARDRAILS

    from .agents import _make_handoff

    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=ToolRegistry(
            tools=[score_projects, validate_project_result]
        ),
        handoffs=[_make_handoff(AGENT_NAME, next_agent)],
        output_guardrails=STAGE_OUTPUT_GUARDRAILS.get(AGENT_NAME, []),
        session=session or Session(),
        graph_factory=create_scoring_graph,
        metadata={
            "score_weights": {
                "function_match": 30,
                "deployment": 20,
                "developer_friendliness": 20,
                "community_activity": 15,
                "documentation": 10,
                "license_friendliness": 5,
            }
        },
    )
