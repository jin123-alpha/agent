"""self_review 节点：DAG 内部逐条审核候选仓库相关性，利用已有精排结果按序检查。"""

import json
import logging
import re

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

REVIEW_PROMPT = """你是一个 GitHub 仓库搜索质检员。根据用户的完整需求，严格判断候选仓库是否相关。

## 用户完整需求
{user_query}

## 候选仓库
- 仓库名: {full_name}
- 描述: {description}
- 语言: {language}
- 星数: {stars}

## 判定规则（严格）
1. **关键词语义命中**：用户需求中的 keywords 至少要有 1 个在**语义上**与仓库功能相关。允许同义词、翻译、近似概念（如 "Web UI" ↔ "browser interface"、"本地部署" ↔ "self-hosted"、"PDF" ↔ "document parsing"）。**不能只检查字面是否出现**，要判断功能是否等价。如果所有关键词语义上都不匹配 → 不通过。
2. **语言匹配**：用户指定了特定编程语言时，仓库语言必须匹配（允许近似，如 JS/TS）。语言不匹配 → 不通过。
3. **排除非项目类**：教程、资料合集、awesome-list、技术博客、面试题、课程项目 → 不通过。
4. **功能方向一致**：仓库的主要用途必须与用户 project_type 描述的方向语义一致，不应只靠名字相似就通过。

## 输出格式
只输出一个 JSON 对象，不要其他内容：
{{"relevant": true/false, "reason": "一句话说明通过或拒绝的核心理由"}}"""


def _hard_precheck(repo: dict, req: dict) -> tuple[bool, str]:
    """硬规则预检：语言不匹配直接拒绝，避免浪费 LLM 调用。"""
    req_language = req.get("language", "")
    if req_language and req_language != "不限":
        repo_lang = (repo.get("language") or "").lower()
        expected = req_language.strip().lower()
        # 允许近似匹配
        js_family = {"javascript", "typescript", "js", "ts"}
        if expected in js_family and repo_lang in js_family:
            return True, ""
        if expected == repo_lang:
            return True, ""
        return False, f"语言不匹配: 要求 {expected}, 实际 {repo_lang}"
    return True, ""


def _check_relevance(
    repo: dict,
    user_query_json: str,
    config: dict,
) -> tuple[bool, str]:
    """用 DeepSeek 判断单个仓库是否与用户需求相关。"""
    # 构建仓库信息
    full_name = repo.get("full_name", "")
    description = repo.get("description", "") or ""
    language = repo.get("language", "") or ""
    stars = repo.get("stars", 0)

    prompt = REVIEW_PROMPT.format(
        user_query=user_query_json,
        full_name=full_name,
        description=description[:500],
        language=language,
        stars=stars,
    )

    llm = ChatOpenAI(
        model=config.get("model", "deepseek-v4-pro"),
        api_key=config.get("api_key", ""),
        base_url=config.get("base_url", ""),
        temperature=0,
        max_tokens=128,
    )

    try:
        response = llm.invoke(prompt)
        output_text = response.content if hasattr(response, "content") else str(response)
    except Exception as exc:
        logger.error(f"self_review: LLM 调用失败 {exc}，默认拒绝")
        return False, f"review failed: {exc}"

    # 解析 JSON
    json_match = re.search(r"\{.*\}", output_text, re.DOTALL)
    if json_match:
        output_text = json_match.group(0)
    try:
        result = json.loads(output_text)
        relevant = result.get("relevant", False)
        reason = result.get("reason", "")
        return relevant, reason
    except json.JSONDecodeError:
        logger.warning(f"self_review: 解析失败，默认拒绝: {output_text[:100]}")
        return False, f"parse failed: {output_text[:100]}"


def self_review(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph 节点：逐条审核过滤后的候选仓库相关性。
    按精排分从高到低依次检查，不相关的跳过，补足到 top_n 条。

    期望 state:
    - filtered_candidates: list[dict] (按 llm_rerank_score 降序)
    - user_query: str (原始用户需求 JSON)

    返回:
    - reviewed_candidates: list[dict] (通过审核的仓库)
    """
    cfg = config or {}
    candidates = state.get("filtered_candidates", [])
    user_query = state.get("user_query", "") or state.get("searchable_query", "")

    # 解析用户需求
    try:
        req = json.loads(user_query) if isinstance(user_query, str) else user_query
        if not isinstance(req, dict):
            req = {}
    except (json.JSONDecodeError, TypeError):
        req = {}

    top_n = req.get("top_n", 5) if isinstance(req, dict) else 5

    reviewed: list[dict] = []
    skipped: list[str] = []

    for i, repo in enumerate(candidates):
        if len(reviewed) >= top_n:
            break

        # 1. 硬规则预检
        pre_ok, pre_reason = _hard_precheck(repo, req)
        if not pre_ok:
            skipped.append(f"{repo.get('full_name', 'N/A')}: {pre_reason} [预检]")
            logger.info(f"self_review precheck fail [{i+1}]: {repo.get('full_name')} — {pre_reason}")
            continue

        # 2. LLM 语义审核
        relevant, reason = _check_relevance(repo, user_query, cfg)

        if relevant:
            reviewed.append(repo)
        else:
            skipped.append(f"{repo.get('full_name', 'N/A')}: {reason}")

    print(f"  [self_review] 审核 {min(len(reviewed) + len(skipped), len(candidates))} 条, "
          f"通过 {len(reviewed)}, 跳过 {len(skipped)}")
    if skipped:
        for s in skipped:
            print(f"    ✗ {s}")

    return {"reviewed_candidates": reviewed}
