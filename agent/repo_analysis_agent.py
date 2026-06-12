"""RepoAnalysisAgent —— 仓库深度分析专家。"""

from handoff import Handoff
from session import Session
from tools import ToolRegistry

from .agent import Agent

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
    )

