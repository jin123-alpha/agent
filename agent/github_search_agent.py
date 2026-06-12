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
    )

