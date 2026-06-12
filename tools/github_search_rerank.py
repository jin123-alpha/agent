"""llm_reranking 节点：用 DeepSeek 单次调用对 top-N 候选仓库精排打分。"""

import json
import logging
import re

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

RERANK_PROMPT = """你是一个 GitHub 仓库搜索精排专家。

## 用户需求
{user_query}

## 候选仓库列表
{candidates_text}

## 任务
根据用户需求与每个候选仓库的相关性，对候选仓库进行精确排序。

## 评分维度
1. 功能匹配度：仓库功能是否满足需求
2. 描述准确性：description 与需求的契合程度

## 输出格式
严格输出 JSON 数组，每个元素包含 full_name 和 relevance_score（0-10 的浮点数），按 relevance_score 降序排列。

```json
[
  {{"full_name": "owner/repo1", "relevance_score": 9.5}},
  {{"full_name": "owner/repo2", "relevance_score": 7.2}},
  ...
]
```

只输出 JSON 数组，不要任何其他文本。"""


def _build_candidates_text(repos: list[dict], max_chars_per_repo: int = 300) -> str:
    """构建候选仓库的文本表示（截断 description 和 combined_doc）。"""
    lines: list[str] = []
    for i, repo in enumerate(repos):
        full_name = repo.get("full_name", "")
        description = repo.get("description", "") or ""
        combined_doc = repo.get("combined_doc", "") or ""

        # 截断
        if len(description) > max_chars_per_repo:
            description = description[:max_chars_per_repo] + "..."
        if len(combined_doc) > max_chars_per_repo:
            combined_doc = combined_doc[:max_chars_per_repo] + "..."

        doc_snippet = combined_doc[:200] if combined_doc else ""

        lines.append(
            f"### {i + 1}. {full_name}\n"
            f"Description: {description}\n"
            f"README 摘要: {doc_snippet}\n"
        )
    return "\n".join(lines)


def _call_deepseek_rerank(
    user_query: str,
    candidates: list[dict],
    config: dict,
) -> list[dict]:
    """调用 DeepSeek 对候选仓库打分排序。"""
    candidates_text = _build_candidates_text(candidates)

    prompt = RERANK_PROMPT.format(
        user_query=user_query,
        candidates_text=candidates_text,
    )

    llm = ChatOpenAI(
        model=config.get("model", "deepseek-v4-pro"),
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        temperature=0,
    )

    print(f"  [rerank] 调用 DeepSeek: model={config.get('model')} base_url={config.get('base_url')}")
    print(f"  [rerank] prompt 长度: {len(prompt)} 字符")
    print(f"  [rerank] 候选数: {len(candidates)}")

    try:
        response = llm.invoke(prompt)
        print(f"  [rerank] DeepSeek 响应长度: {len(response.content) if hasattr(response, 'content') else len(str(response))}")
        output_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        print(f"  [rerank] DeepSeek 调用失败: {type(exc).__name__}: {exc}")
        logger.error(f"llm_reranking: DeepSeek 调用失败 ({exc})，使用语义检索顺序")
        return candidates

    # 提取 JSON 数组
    json_match = re.search(r"```(?:json)?\s*(.*?)\s*```", output_text, re.DOTALL)
    if json_match:
        output_text = json_match.group(1).strip()

    try:
        scored = json.loads(output_text)
        if not isinstance(scored, list):
            logger.warning("llm_reranking: LLM 返回不是数组，回退到原始顺序")
            return candidates
        print(f"  [rerank] 成功解析 {len(scored)} 个评分")
        return scored
    except json.JSONDecodeError:
        print(f"  [rerank] JSON 解析失败，原始响应前200字符: {output_text[:200]}")
        logger.warning("llm_reranking: LLM 返回解析失败，回退到原始顺序")
        return candidates


def _apply_rerank_scores(
    candidates: list[dict], scored: list[dict]
) -> list[dict]:
    """将 LLM 精排分数合并到候选仓库，按分数降序排列。"""
    score_map: dict[str, float] = {}
    for item in scored:
        if isinstance(item, dict):
            full_name = item.get("full_name", "")
            score = item.get("relevance_score", 0)
            try:
                score_map[full_name] = float(score)
            except (TypeError, ValueError):
                score_map[full_name] = 0.0

    for repo in candidates:
        fn = repo.get("full_name", "")
        if fn in score_map:
            repo["llm_rerank_score"] = score_map[fn]
        else:
            # 未出现在 LLM 输出中，保留 semantic_similarity 作为分数
            repo["llm_rerank_score"] = repo.get("semantic_similarity", 0)

    return sorted(
        candidates,
        key=lambda x: x.get("llm_rerank_score", 0),
        reverse=True,
    )


def llm_reranking(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph 节点：用 DeepSeek 对 semantic_ranked 的 top-N 精排。

    期望 state:
    - user_query: str
    - semantic_ranked: list[dict]

    返回:
    - reranked_candidates: list[dict] (附 llm_rerank_score，降序)
    """
    cfg = config or {}
    rerank_top_n = cfg.get("llm_rerank_top_n", 50)
    semantic_ranked = state.get("semantic_ranked", [])

    if not semantic_ranked:
        logger.warning("llm_reranking: semantic_ranked 为空")
        return {"reranked_candidates": []}

    user_query = state.get("user_query", "") or state.get("searchable_query", "")

    # 取 top-N 送入 LLM 精排
    candidates_for_rerank = semantic_ranked[:rerank_top_n]

    try:
        scored = _call_deepseek_rerank(user_query, candidates_for_rerank, cfg)
        reranked = _apply_rerank_scores(candidates_for_rerank, scored)

        # 未参与精排的保持原序追加
        if len(semantic_ranked) > rerank_top_n:
            reranked.extend(semantic_ranked[rerank_top_n:])

        logger.info(
            f"llm_reranking: {len(reranked)} candidates after reranking "
            f"(top score: {reranked[0].get('llm_rerank_score', 0):.2f} "
            f"if results)"
        )
    except Exception as exc:
        logger.error(f"llm_reranking: DeepSeek 调用失败 ({exc})，使用语义检索顺序")
        for repo in semantic_ranked:
            repo["llm_rerank_score"] = repo.get("semantic_similarity", 0)
        reranked = sorted(
            semantic_ranked,
            key=lambda x: x.get("semantic_similarity", 0),
            reverse=True,
        )

    return {"reranked_candidates": reranked}
