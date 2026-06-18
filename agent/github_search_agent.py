"""GitHubSearchAgent —— GitHub 仓库搜索专家。

DAG 流水线:
  START → convert_query → ingest_github_repos → dense_retrieval
        → llm_reranking → threshold_filtering → finalize → END
"""

import json
import logging

from handoff import Handoff
from session import Session
from tools import ToolRegistry

from .agent import Agent

logger = logging.getLogger(__name__)

AGENT_NAME = "GitHubSearchAgent"

INSTRUCTIONS = """\
你是 GitHubSearchAgent —— GitHub 仓库搜索专家。

## 工作流程
你通过多阶段搜索流水线工作：
1. 将结构化需求转为 GitHub 搜索查询
2. 调用 GitHub Search API 拉取仓库并抓取文档
3. 用语义检索 + BM25 混合初排
4. 用 LLM 精排 top-N 候选
5. 过滤低质量结果

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
"""


def _create_dag_graph(**kwargs):
    """构建 GitHubSearchAgent 的 DAG 流水线 LangGraph 图。"""
    from runner.langgraph_dependencies import import_langgraph_dependencies

    deps = import_langgraph_dependencies()
    HumanMessage = deps["HumanMessage"]
    AIMessage = deps["AIMessage"]
    END = deps["END"]
    START = deps["START"]
    StateGraph = deps["StateGraph"]
    add_messages = deps["add_messages"]
    Annotated = deps["Annotated"]
    TypedDict = deps["TypedDict"]

    runtime_config = kwargs.get("config", {})
    on_status = runtime_config.get("_runtime_status_callback")
    config_from_runner = {
        key: value
        for key, value in runtime_config.items()
        if key != "_runtime_status_callback"
    }

    def report_status(message: str) -> None:
        if callable(on_status):
            on_status(message)

    # ── State 定义 ──────────────────────────────────────────────
    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        llm_calls: int
        searchable_query: str
        target_language: str
        user_query: str
        repositories: list[dict]
        semantic_ranked: list[dict]
        reranked_candidates: list[dict]
        filtered_candidates: list[dict]
        reviewed_candidates: list[dict]
        _dag_config: dict  # 内部使用：传递 runner config 到各节点

    # ── 导入节点函数 ────────────────────────────────────────────
    from tools.github_search_query import convert_query
    from tools.github_search_ingest import ingest_github_repos
    from tools.github_search_retrieval import dense_retrieval
    from tools.github_search_rerank import llm_reranking
    from tools.github_search_filter import threshold_filtering
    from tools.github_search_self_review import self_review

    # ── 包装器：将 config 冻结到 state 中供下游节点读取 ──────────

    def _convert_query(state: AgentState) -> dict:
        report_status("正在解析搜索需求")
        result = convert_query(state, config_from_runner)
        result["_dag_config"] = config_from_runner
        logger.debug(
            "convert_query completed: %s",
            result.get("searchable_query", ""),
        )
        return result

    def _ingest_github_repos(state: AgentState) -> dict:
        report_status("正在搜索 GitHub 并抓取仓库文档")
        cfg = state.get("_dag_config", config_from_runner)
        result = ingest_github_repos(state, cfg, report=report_status)
        repos = result.get("repositories", [])
        logger.debug("ingest_github_repos completed: %d repositories", len(repos))
        return result

    def _dense_retrieval(state: AgentState) -> dict:
        report_status("正在进行向量检索与 BM25 语义排序")
        cfg = state.get("_dag_config", config_from_runner)
        result = dense_retrieval(state, cfg, report=report_status)
        ranked = result.get("semantic_ranked", [])
        logger.debug("dense_retrieval completed: %d candidates", len(ranked))
        return result

    def _llm_reranking(state: AgentState) -> dict:
        report_status("正在使用 LLM 精排候选项目")
        cfg = state.get("_dag_config", config_from_runner)
        result = llm_reranking(state, cfg, report=report_status)
        reranked = result.get("reranked_candidates", [])
        logger.debug("llm_reranking completed: %d candidates", len(reranked))
        return result

    def _threshold_filtering(state: AgentState) -> dict:
        report_status("正在按质量阈值过滤候选项目")
        cfg = state.get("_dag_config", config_from_runner)
        result = threshold_filtering(state, cfg, report=report_status)
        filtered = result.get("filtered_candidates", [])
        logger.debug("threshold_filtering completed: %d candidates", len(filtered))
        return result

    def _self_review(state: AgentState) -> dict:
        report_status("正在审核候选项目相关性")
        cfg = state.get("_dag_config", config_from_runner)
        result = self_review(state, cfg, report=report_status)
        reviewed = result.get("reviewed_candidates", [])
        logger.debug("self_review completed: %d candidates", len(reviewed))
        return result

    # ── finalize 节点：输出 JSON → AIMessage ─────────────────────
    def _finalize(state: AgentState) -> dict:
        report_status("正在整理 GitHub 搜索结果")
        filtered = state.get("reviewed_candidates", []) or state.get("filtered_candidates", [])
        user_query = state.get("user_query", "") or state.get("searchable_query", "")

        # 尝试从 user_query 中提取 top_n
        top_n = 5
        try:
            if user_query:
                req = json.loads(user_query) if isinstance(user_query, str) else user_query
                if isinstance(req, dict) and isinstance(req.get("top_n"), int):
                    top_n = req["top_n"]
        except (json.JSONDecodeError, TypeError):
            pass

        # 截断到 top_n
        results = filtered[:top_n]

        # 只输出版本信息
        output = []
        for repo in results:
            output.append(
                {
                    "full_name": repo.get("full_name", ""),
                    "html_url": repo.get("html_url", ""),
                    "description": repo.get("description", ""),
                    "stars": repo.get("stars", 0),
                    "language": repo.get("language", ""),
                    "license": repo.get("license", ""),
                    "updated_at": repo.get("updated_at", ""),
                }
            )

        output_json = json.dumps(output, ensure_ascii=False)
        logger.debug("finalize completed: top_%d, %d repositories", top_n, len(output))

        return {
            "messages": [AIMessage(content=output_json)],
            "llm_calls": state.get("llm_calls", 0),
        }

    # ── 构建图 ──────────────────────────────────────────────────
    builder = StateGraph(AgentState)
    builder.add_node("convert_query", _convert_query)
    builder.add_node("ingest_github_repos", _ingest_github_repos)
    builder.add_node("dense_retrieval", _dense_retrieval)
    builder.add_node("llm_reranking", _llm_reranking)
    builder.add_node("threshold_filtering", _threshold_filtering)
    builder.add_node("self_review", _self_review)
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "convert_query")
    builder.add_edge("convert_query", "ingest_github_repos")
    builder.add_edge("ingest_github_repos", "dense_retrieval")
    builder.add_edge("dense_retrieval", "llm_reranking")
    builder.add_edge("llm_reranking", "threshold_filtering")
    builder.add_edge("threshold_filtering", "self_review")
    builder.add_edge("self_review", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile(), HumanMessage


def create_github_search_agent(
    session: Session | None = None,
    next_agent: str = "RepoAnalysisAgent",
) -> Agent:
    """创建 GitHub 搜索 Agent。"""
    from guardrail.output_guardrails import STAGE_OUTPUT_GUARDRAILS
    from .agents import _make_handoff

    return Agent(
        name=AGENT_NAME,
        instructions=INSTRUCTIONS,
        tools=ToolRegistry(tools=[]),
        handoffs=[_make_handoff(AGENT_NAME, next_agent)],
        output_guardrails=STAGE_OUTPUT_GUARDRAILS.get(AGENT_NAME, []),
        session=session or Session(),
        graph_factory=_create_dag_graph,
    )
