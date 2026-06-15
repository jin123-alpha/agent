from __future__ import annotations

"""dense_retrieval 节点：SentenceTransformer + BM25 混合语义检索。"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 全局缓存的模型实例（懒加载）
_sem_model = None


def _get_sem_model():
    """懒加载 SentenceTransformer 模型（优先使用本地缓存）。"""
    global _sem_model
    if _sem_model is None:
        from pathlib import Path

        from sentence_transformers import SentenceTransformer

        # 优先使用本地下载的模型，否则从 HF 拉取
        local_path = Path(__file__).resolve().parents[1] / "all-mpnet-base-v2"
        if local_path.exists():
            model_name = str(local_path)
            logger.info(f"Loading SentenceTransformer from local: {model_name}")
        else:
            model_name = "all-mpnet-base-v2"
            logger.info(f"Loading SentenceTransformer from HF: {model_name}")
        _sem_model = SentenceTransformer(model_name)
    return _sem_model


def _normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    """L2 归一化。"""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)


def _compute_dense_scores(
    query: str, docs: list[str]
) -> np.ndarray:
    """用 SentenceTransformer 编码 query 和 docs，返回余弦相似度分数。"""
    import faiss

    model = _get_sem_model()
    doc_embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=False, batch_size=16)
    doc_embeddings = _normalize_embeddings(doc_embeddings)

    query_embedding = model.encode(query, convert_to_numpy=True)
    query_embedding = _normalize_embeddings(np.expand_dims(query_embedding, axis=0))[0]

    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(doc_embeddings)
    k = min(doc_embeddings.shape[0], 1) if doc_embeddings.shape[0] > 0 else 1
    D, I = index.search(np.expand_dims(query_embedding, axis=0), k)

    # D, I 是 (1, k) 形状 — 但我们需要一个 n_docs 长度的数组
    # 把内积分数映射回每个文档
    scores = np.zeros(len(docs))
    for idx, score in zip(I[0], D[0]):
        scores[idx] = float(score)

    return scores


def _compute_bm25_scores(query: str, docs: list[str]) -> np.ndarray:
    """用 BM25 计算稀疏检索分数。"""
    from rank_bm25 import BM25Okapi

    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    query_tokens = query.split()
    scores = np.array(bm25.get_scores(query_tokens))
    return scores


def _min_max_norm(values: np.ndarray) -> np.ndarray:
    """min-max 归一化到 [0, 1]。"""
    v_min = values.min()
    v_max = values.max()
    if v_max - v_min < 1e-10:
        return np.full_like(values, 0.5)
    return (values - v_min) / (v_max - v_min)


def dense_retrieval(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph 节点：混合语义检索 (SentenceTransformer + BM25)。

    期望 state:
    - user_query: str (原始结构化 JSON 或自然语言)
    - repositories: list[dict] (含 combined_doc)

    返回:
    - semantic_ranked: list[dict] (按 semantic_similarity 降序排列)
    """
    cfg = config or {}
    alpha = cfg.get("retrieval_alpha", 0.7)
    dense_retrieval_k = cfg.get("dense_retrieval_k", 100)

    repos = state.get("repositories", [])
    if not repos:
        logger.warning("dense_retrieval: 没有仓库数据")
        return {"semantic_ranked": []}

    # 截断文档到 1000 字符（减少编码量）
    docs = []
    for repo in repos:
        doc = repo.get("combined_doc", "") or repo.get("description", "")
        docs.append(doc[:1000] if len(doc) > 1000 else doc)
    logger.debug(f"dense_retrieval: {len(docs)} docs, total chars={sum(len(d) for d in docs)}")
    user_query = state.get("user_query", "") or state.get("searchable_query", "")

    if not user_query.strip():
        logger.warning("dense_retrieval: user_query 为空，跳过语义检索")
        state["semantic_ranked"] = repos
        return {"semantic_ranked": repos}

    try:
        dense_scores = _compute_dense_scores(user_query, docs)
        norm_dense = _min_max_norm(dense_scores)
    except Exception as exc:
        logger.error(f"dense_retrieval: SentenceTransformer 编码失败 ({exc})，降级到纯 BM25")
        norm_dense = np.zeros(len(docs))

    try:
        bm25_scores = _compute_bm25_scores(user_query, docs)
        norm_bm25 = _min_max_norm(bm25_scores)
    except Exception as exc:
        logger.error(f"dense_retrieval: BM25 计算失败 ({exc})，降级到纯 Dense")
        norm_bm25 = np.zeros(len(docs))

    # 混合：alpha * dense + (1-alpha) * bm25
    combined = alpha * norm_dense + (1 - alpha) * norm_bm25

    for idx, repo in enumerate(repos):
        repo["semantic_similarity"] = float(combined[idx])

    semantic_ranked = sorted(
        repos,
        key=lambda x: x.get("semantic_similarity", 0),
        reverse=True,
    )

    # 截断到 top-K
    if dense_retrieval_k > 0 and len(semantic_ranked) > dense_retrieval_k:
        semantic_ranked = semantic_ranked[:dense_retrieval_k]

    logger.info(
        f"dense_retrieval: {len(semantic_ranked)} candidates ranked "
        f"(alpha={alpha}, dense_mean={norm_dense.mean():.4f}, bm25_mean={norm_bm25.mean():.4f})"
    )

    return {"semantic_ranked": semantic_ranked}
