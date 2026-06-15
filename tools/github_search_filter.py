from __future__ import annotations

"""threshold_filtering 节点：过滤低 star + 低精排分的仓库。"""

import logging

logger = logging.getLogger(__name__)


def threshold_filtering(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph 节点：过滤同时满足「低星数」且「低精排分」的仓库。

    规则：同时满足 stars < min_stars 且 llm_rerank_score < rerank_threshold 则丢弃。
    fallback：如果过滤后为空，保留全部 reranked_candidates。

    期望 state:
    - reranked_candidates: list[dict] (含 stars, llm_rerank_score)

    返回:
    - filtered_candidates: list[dict]
    """
    cfg = config or {}
    min_stars = cfg.get("min_stars", 50)
    rerank_threshold = cfg.get("rerank_threshold", 5.5)

    reranked = state.get("reranked_candidates", [])
    if not reranked:
        logger.warning("threshold_filtering: reranked_candidates 为空")
        return {"filtered_candidates": []}

    filtered: list[dict] = []
    for repo in reranked:
        stars = repo.get("stars", 0)
        score = repo.get("llm_rerank_score", 0)
        # 同时满足两个阈值才丢弃
        if stars < min_stars and score < rerank_threshold:
            logger.debug(
                f"filtering out {repo.get('full_name')}: "
                f"stars={stars} < {min_stars}, score={score:.2f} < {rerank_threshold}"
            )
            continue
        filtered.append(repo)

    # fallback: 如果全部被过滤掉，保留所有
    if not filtered:
        logger.warning(
            f"threshold_filtering: 所有 {len(reranked)} 个仓库被过滤，回退到全部保留"
        )
        filtered = list(reranked)

    logger.info(
        f"threshold_filtering: {len(filtered)}/{len(reranked)} candidates remain "
        f"(min_stars={min_stars}, rerank_threshold={rerank_threshold})"
    )

    return {"filtered_candidates": filtered}
