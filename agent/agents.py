"""
Agent 流水线编排模块。

各 Agent 定义在独立文件中，本模块负责：
1. 定义流水线顺序常量
2. 提供 handoff 工厂函数（供各 Agent 文件调用）
3. 汇总导出 create_all_agents()
"""

from handoff import Handoff
from session import Session

from .agent import Agent
from .github_search_agent import create_github_search_agent
from .repo_analysis_agent import create_repo_analysis_agent
from .report_agent import create_report_agent
from .requirement_agent import create_requirement_agent
from .scoring_agent import create_scoring_agent

# ---------------------------------------------------------------------------
# Agent names（流水线顺序）
# ---------------------------------------------------------------------------

AGENT_REQUIREMENT = "RequirementAgent"
AGENT_GITHUB_SEARCH = "GitHubSearchAgent"
AGENT_REPO_ANALYSIS = "RepoAnalysisAgent"
AGENT_SCORING = "ScoringAgent"
AGENT_REPORT = "ReportAgent"

AGENT_PIPELINE = [
    AGENT_REQUIREMENT,
    AGENT_GITHUB_SEARCH,
    AGENT_REPO_ANALYSIS,
    AGENT_SCORING,
    AGENT_REPORT,
]

# ---------------------------------------------------------------------------
# Handoff 工厂（供各 Agent 文件调用）
# ---------------------------------------------------------------------------


def _always_handoff(_user_input: str) -> bool:
    """流水线中始终触发 handoff 到下一个 Agent。"""
    return True


def _make_handoff(current_name: str, target_name: str) -> Handoff:
    return Handoff(
        name=f"{current_name}_to_{target_name}",
        can_handle=_always_handoff,
        target_agent=target_name,
    )


# ---------------------------------------------------------------------------
# 汇总
# ---------------------------------------------------------------------------

AGENT_FACTORIES = {
    AGENT_REQUIREMENT: create_requirement_agent,
    AGENT_GITHUB_SEARCH: create_github_search_agent,
    AGENT_REPO_ANALYSIS: create_repo_analysis_agent,
    AGENT_SCORING: create_scoring_agent,
    AGENT_REPORT: create_report_agent,
}


def create_all_agents(session: Session | None = None) -> dict[str, Agent]:
    """创建全部 5 个 Agent 并返回 {name: Agent} 字典。"""
    shared_session = session or Session()
    return {name: factory(shared_session) for name, factory in AGENT_FACTORIES.items()}
