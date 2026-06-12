"""
各阶段输出 Guardrail —— 纯规则校验（不调用 LLM）。

负责快速拦截格式错误、字段缺失、范围越界等问题，
通过后再交给 CriticAgent 做语义层审查。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .guardrail import OutputGuardrail, OutputGuardrailResult


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _try_parse_json_array(text: str) -> tuple[list | None, str]:
    """尝试解析 JSON 数组，支持 markdown 代码块包裹。返回 (parsed, error)。"""
    cleaned = text.strip()
    # 去掉 markdown 代码块
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        json_lines = []
        in_block = False
        for line in lines:
            if line.strip().startswith("```") and not in_block:
                in_block = True
                continue
            elif line.strip().startswith("```") and in_block:
                break
            elif in_block:
                json_lines.append(line)
        cleaned = "\n".join(json_lines).strip()

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return None, "JSON 解析成功但不是数组类型"
        return parsed, ""
    except json.JSONDecodeError as e:
        return None, f"JSON 解析失败: {e}"


def _check_required_fields(item: dict, fields: list[str]) -> list[str]:
    """检查字典是否包含所有必要字段，返回缺失字段列表。"""
    return [f for f in fields if f not in item]


# ---------------------------------------------------------------------------
# GitHubSearchAgent 输出 Guardrail
# ---------------------------------------------------------------------------

SEARCH_REQUIRED_FIELDS = ["full_name", "stars", "language"]


def _check_search_output(output: str, context: dict[str, Any]) -> OutputGuardrailResult:
    """GitHubSearchAgent 纯规则校验。"""
    failures = []

    # 1. 非空
    if not output or not output.strip():
        return OutputGuardrailResult(ok=False, failures=["搜索结果为空"])

    # 2. JSON 格式
    parsed, err = _try_parse_json_array(output)
    if parsed is None:
        return OutputGuardrailResult(ok=False, failures=[f"输出不是合法 JSON 数组: {err}"])

    # 3. 非空数组
    if len(parsed) == 0:
        failures.append("搜索结果数组为空，未返回任何仓库")

    # 4. 必要字段
    for i, repo in enumerate(parsed):
        if not isinstance(repo, dict):
            failures.append(f"第 {i+1} 项不是 JSON 对象")
            continue
        missing = _check_required_fields(repo, SEARCH_REQUIRED_FIELDS)
        if missing:
            failures.append(f"第 {i+1} 项 ({repo.get('full_name', '?')}) 缺少字段: {missing}")

    # 5. stars 数值校验
    for i, repo in enumerate(parsed):
        if not isinstance(repo, dict):
            continue
        stars = repo.get("stars")
        if stars is not None and not isinstance(stars, (int, float)):
            failures.append(f"第 {i+1} 项 stars 不是数值: {stars}")

    return OutputGuardrailResult(ok=len(failures) == 0, failures=failures)


SEARCH_OUTPUT_GUARDRAIL = OutputGuardrail(
    name="search_output_format",
    check=_check_search_output,
)


# ---------------------------------------------------------------------------
# RepoAnalysisAgent 输出 Guardrail
# ---------------------------------------------------------------------------

ANALYSIS_REQUIRED_FIELDS = [
    "full_name", "readme_summary", "features",
    "has_tests", "has_docs",
]


def _check_analysis_output(output: str, context: dict[str, Any]) -> OutputGuardrailResult:
    """RepoAnalysisAgent 纯规则校验。"""
    failures = []

    if not output or not output.strip():
        return OutputGuardrailResult(ok=False, failures=["分析结果为空"])

    parsed, err = _try_parse_json_array(output)
    if parsed is None:
        return OutputGuardrailResult(ok=False, failures=[f"输出不是合法 JSON 数组: {err}"])

    if len(parsed) == 0:
        failures.append("分析结果数组为空")

    # 覆盖率检查（与搜索结果数量对比）
    search_output = context.get("GitHubSearchAgent", "")
    if search_output:
        search_parsed, _ = _try_parse_json_array(search_output)
        if search_parsed and len(parsed) != len(search_parsed):
            failures.append(
                f"分析结果数量 ({len(parsed)}) 与搜索结果数量 ({len(search_parsed)}) 不一致"
            )

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            failures.append(f"第 {i+1} 项不是 JSON 对象")
            continue
        missing = _check_required_fields(item, ANALYSIS_REQUIRED_FIELDS)
        if missing:
            failures.append(f"第 {i+1} 项 ({item.get('full_name', '?')}) 缺少字段: {missing}")

        # readme_summary 非空
        if item.get("readme_summary") is not None and not str(item["readme_summary"]).strip():
            failures.append(f"第 {i+1} 项 readme_summary 为空字符串")

    return OutputGuardrailResult(ok=len(failures) == 0, failures=failures)


ANALYSIS_OUTPUT_GUARDRAIL = OutputGuardrail(
    name="analysis_output_format",
    check=_check_analysis_output,
)


# ---------------------------------------------------------------------------
# ScoringAgent 输出 Guardrail
# ---------------------------------------------------------------------------

SCORING_REQUIRED_FIELDS = ["full_name", "scores", "weighted_total", "rank"]
SCORE_DIMENSIONS = [
    "requirement_match", "community_activity",
    "doc_quality", "engineering", "deployment",
]


def _check_scoring_output(output: str, context: dict[str, Any]) -> OutputGuardrailResult:
    """ScoringAgent 纯规则校验。"""
    failures = []

    if not output or not output.strip():
        return OutputGuardrailResult(ok=False, failures=["评分结果为空"])

    parsed, err = _try_parse_json_array(output)
    if parsed is None:
        return OutputGuardrailResult(ok=False, failures=[f"输出不是合法 JSON 数组: {err}"])

    if len(parsed) == 0:
        failures.append("评分结果数组为空")

    for i, item in enumerate(parsed):
        if not isinstance(item, dict):
            failures.append(f"第 {i+1} 项不是 JSON 对象")
            continue

        missing = _check_required_fields(item, SCORING_REQUIRED_FIELDS)
        if missing:
            failures.append(f"第 {i+1} 项缺少字段: {missing}")
            continue

        # scores 子字段
        scores = item.get("scores", {})
        if isinstance(scores, dict):
            missing_dims = [d for d in SCORE_DIMENSIONS if d not in scores]
            if missing_dims:
                failures.append(f"第 {i+1} 项 scores 缺少维度: {missing_dims}")

            # 分数范围 0-10
            for dim, val in scores.items():
                if isinstance(val, (int, float)) and (val < 0 or val > 10):
                    failures.append(f"第 {i+1} 项 {dim}={val} 不在 0-10 范围内")
        else:
            failures.append(f"第 {i+1} 项 scores 不是 JSON 对象")

    # 排序检查
    if len(parsed) >= 2:
        totals = []
        for item in parsed:
            if isinstance(item, dict):
                wt = item.get("weighted_total")
                if isinstance(wt, (int, float)):
                    totals.append(wt)
        if totals and totals != sorted(totals, reverse=True):
            failures.append("weighted_total 未按降序排列")

    return OutputGuardrailResult(ok=len(failures) == 0, failures=failures)


SCORING_OUTPUT_GUARDRAIL = OutputGuardrail(
    name="scoring_output_format",
    check=_check_scoring_output,
)


# ---------------------------------------------------------------------------
# ReportAgent 输出 Guardrail
# ---------------------------------------------------------------------------

# 报告结构必须包含的关键标记（宽松匹配）
REPORT_REQUIRED_PATTERNS = [
    (r"需求概述|需求分析|项目需求", "缺少'需求概述'章节"),
    (r"候选项目|项目概览|项目对比", "缺少'候选项目概览'章节"),
    (r"\|.*\|.*\|", "缺少项目对比表格"),
    (r"详细分析|项目分析|深度分析", "缺少'详细分析'章节"),
    (r"推荐结论|总结|推荐", "缺少'推荐结论'章节"),
]


def _check_report_output(output: str, context: dict[str, Any]) -> OutputGuardrailResult:
    """ReportAgent 纯规则校验。"""
    failures = []

    if not output or not output.strip():
        return OutputGuardrailResult(ok=False, failures=["报告内容为空"])

    if len(output.strip()) < 200:
        failures.append(f"报告过短 ({len(output.strip())} 字符)，可能内容不完整")

    for pattern, err_msg in REPORT_REQUIRED_PATTERNS:
        if not re.search(pattern, output):
            failures.append(err_msg)

    return OutputGuardrailResult(ok=len(failures) == 0, failures=failures)


REPORT_OUTPUT_GUARDRAIL = OutputGuardrail(
    name="report_output_format",
    check=_check_report_output,
)


# ---------------------------------------------------------------------------
# 按阶段名索引
# ---------------------------------------------------------------------------

STAGE_OUTPUT_GUARDRAILS: dict[str, list[OutputGuardrail]] = {
    "GitHubSearchAgent": [SEARCH_OUTPUT_GUARDRAIL],
    "RepoAnalysisAgent": [ANALYSIS_OUTPUT_GUARDRAIL],
    "ScoringAgent": [SCORING_OUTPUT_GUARDRAIL],
    "ReportAgent": [REPORT_OUTPUT_GUARDRAIL],
}
