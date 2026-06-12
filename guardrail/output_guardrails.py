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
from tools.scoring_tools import validate_project_result, validate_report


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _try_parse_json_array(text: str) -> tuple[list | None, str]:
    """尝试解析 JSON 数组，支持 markdown 代码块包裹。返回 (parsed, error)。"""
    cleaned = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.DOTALL)
    if fenced_match:
        cleaned = fenced_match.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        if not isinstance(parsed, list):
            return None, "JSON 解析成功但不是数组类型"
        return parsed, ""
    except json.JSONDecodeError as e:
        array_match = re.search(r"\[[\s\S]*\]", cleaned)
        if array_match:
            try:
                parsed = json.loads(array_match.group(0))
                if not isinstance(parsed, list):
                    return None, "JSON 解析成功但不是数组类型"
                return parsed, ""
            except json.JSONDecodeError:
                pass
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

SCORING_REQUIRED_FIELDS = [
    "project", "total_score", "scores", "reason", "evidence", "rank",
]
SCORE_DIMENSIONS = [
    "function_match", "deployment", "developer_friendliness",
    "community_activity", "documentation", "license_friendliness",
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

    source_projects = context.get("RepoAnalysisAgent", [])
    validation = json.loads(validate_project_result(parsed, source_projects))
    failures.extend(validation["errors"])

    return OutputGuardrailResult(ok=len(failures) == 0, failures=failures)


SCORING_OUTPUT_GUARDRAIL = OutputGuardrail(
    name="scoring_output_format",
    check=_check_scoring_output,
)


# ---------------------------------------------------------------------------
# ReportAgent 输出 Guardrail
# ---------------------------------------------------------------------------

def _check_report_output(output: str, context: dict[str, Any]) -> OutputGuardrailResult:
    """ReportAgent 纯规则校验。"""
    failures = []

    if not output or not output.strip():
        return OutputGuardrailResult(ok=False, failures=["报告内容为空"])

    known_projects = context.get("GitHubSearchAgent", [])
    validation = json.loads(validate_report(output, known_projects))
    failures.extend(validation["errors"])

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
