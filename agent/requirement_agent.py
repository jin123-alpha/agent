"""RequirementAgent —— 需求解析专家。"""

from handoff import Handoff
from session import Session
from tools import ToolRegistry

from .agent import Agent

AGENT_NAME = "RequirementAgent"

INSTRUCTIONS = """\
你是 RequirementAgent —— 需求解析专家。

## 任务
接收用户的技术选型需求（自然语言），输出一份结构化 JSON。

## 输出格式（严格 JSON）
{
  "project_type": "需求所属的项目类型，如 web框架/ORM/消息队列",
  "keywords": ["关键词1", "关键词2"],
  "language": "偏好语言，如 Python / Go / 不限",
  "constraints": ["约束条件，如 必须支持 PostgreSQL"],
  "preferences": ["偏好，如 活跃社区、文档完善"],
  "top_n": 5
}

## 规则
- 如果用户未指定语言，设为 "不限"。
- 如果用户未指定数量，默认 top_n=5。
- 只输出 JSON，不要多余解释。
- 分析完成后，将解析结果交给下一个 Agent。
"""


def create_requirement_agent(
    session: Session | None = None,
    next_agent: str = "GitHubSearchAgent",
) -> Agent:
    """创建需求解析 Agent。"""
    from .agents import _make_handoff

    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=ToolRegistry(tools=[]),
        handoffs=[_make_handoff(AGENT_NAME, next_agent)],
        session=session or Session(),
    )
