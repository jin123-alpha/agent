"""
Agent 流水线编排模块。

各 Agent 定义在独立文件中，本模块负责：
1. 定义流水线顺序常量
2. 提供 handoff 工厂函数（供各 Agent 文件调用）
3. 汇总导出 create_all_agents()

流水线（含 Critic）：
Requirement → GitHubSearch → Critic(search) → RepoAnalysis → Critic(analysis)
→ Scoring → Report → Critic(report)
"""

from handoff import Handoff
from session import Session

from .agent import Agent
from .critic_agent import STAGE_CHECKLISTS, create_critic_agent
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
AGENT_CRITIC = "CriticAgent"

# Critic 阶段名 = CriticAgent_{被审查Agent名}
AGENT_CRITIC_SEARCH = f"{AGENT_CRITIC}_{AGENT_GITHUB_SEARCH}"
AGENT_CRITIC_ANALYSIS = f"{AGENT_CRITIC}_{AGENT_REPO_ANALYSIS}"
AGENT_CRITIC_SCORING = f"{AGENT_CRITIC}_{AGENT_SCORING}"
AGENT_CRITIC_REPORT = f"{AGENT_CRITIC}_{AGENT_REPORT}"

# 需要 Critic 审查的阶段
CRITIC_STAGES = [
    AGENT_GITHUB_SEARCH,
    AGENT_REPO_ANALYSIS,
    AGENT_SCORING,
    AGENT_REPORT,
]

# 完整流水线（含 Critic 步骤）
AGENT_PIPELINE = [
    AGENT_REQUIREMENT,
    AGENT_GITHUB_SEARCH,
    AGENT_CRITIC_SEARCH,      # ← Critic 审查搜索结果
    AGENT_REPO_ANALYSIS,
    AGENT_CRITIC_ANALYSIS,    # ← Critic 审查分析结果
    AGENT_SCORING,
    AGENT_CRITIC_SCORING,     # ← Critic 审查评分证据与语义一致性
    AGENT_REPORT,
    AGENT_CRITIC_REPORT,      # ← Critic 审查报告
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
    """创建全部 Agent（含 Critic）并返回 {name: Agent} 字典。"""
    shared_session = session or Session()

    agents = {
        name: factory(shared_session)
        for name, factory in AGENT_FACTORIES.items()
    }

    # 为每个需要审查的阶段创建 CriticAgent
    for stage in CRITIC_STAGES:
        critic = create_critic_agent(stage, session=shared_session)
        agents[critic.name] = critic

    return agents
