"""Deterministic project scoring and quality validation tools."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any


SCORE_WEIGHTS = {
    "function_match": 30,
    "deployment": 20,
    "developer_friendliness": 20,
    "community_activity": 15,
    "documentation": 10,
    "license_friendliness": 5,
}

REPORT_SECTIONS = [
    "用户需求分析",
    "候选项目概览",
    "GitHub 基础信息对比",
    "功能特性对比",
    "项目评分矩阵",
    "推荐方案",
    "风险分析",
    "后续开发路线建议",
    "参考来源",
]

FRIENDLY_LICENSES = {
    "MIT",
    "APACHE-2.0",
    "APACHE 2.0",
    "BSD-2-CLAUSE",
    "BSD-3-CLAUSE",
    "ISC",
    "MPL-2.0",
}


def _json_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _as_list(value: Any) -> list[Any]:
    parsed = _json_value(value)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    parsed = _json_value(value)
    return parsed if isinstance(parsed, dict) else {}


def _normalise_text(value: Any) -> str:
    if isinstance(value, (list, tuple, set)):
        return " ".join(_normalise_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_normalise_text(item)}" for key, item in value.items())
    return str(value or "").lower()


def _project_name(project: dict[str, Any]) -> str:
    return str(
        project.get("project")
        or project.get("full_name")
        or project.get("name")
        or ""
    ).strip()


def _merge_project_data(
    projects: list[Any],
    metadata: list[Any],
) -> list[dict[str, Any]]:
    metadata_by_name = {
        _project_name(item): item
        for item in metadata
        if isinstance(item, dict) and _project_name(item)
    }
    merged: list[dict[str, Any]] = []
    for item in projects:
        if not isinstance(item, dict):
            continue
        name = _project_name(item)
        combined = dict(metadata_by_name.get(name, {}))
        combined.update(item)
        merged.append(combined)
    return merged


def _keyword_tokens(requirements: dict[str, Any]) -> list[str]:
    values = [
        requirements.get("project_type", ""),
        requirements.get("keywords", []),
        requirements.get("constraints", []),
    ]
    tokens: list[str] = []
    for token in re.findall(r"[\w.+#-]{2,}", _normalise_text(values)):
        if token not in tokens:
            tokens.append(token)
    return tokens


def _boolean_score(project: dict[str, Any], fields: list[str], maximum: int) -> float:
    if not fields:
        return 0.0
    hits = sum(bool(project.get(field)) for field in fields)
    return maximum * hits / len(fields)


def _function_match(project: dict[str, Any], requirements: dict[str, Any]) -> tuple[int, str]:
    tokens = _keyword_tokens(requirements)
    if not tokens:
        return 15, "需求未提供可比较关键词，按中性分计。"

    project_text = _normalise_text(
        [
            project.get("description"),
            project.get("purpose"),
            project.get("readme_summary"),
            project.get("features"),
            project.get("main_features"),
        ]
    )
    matched = [token for token in tokens if token in project_text]
    ratio = len(matched) / len(tokens)
    score = round(30 * ratio)
    reason = f"匹配 {len(matched)}/{len(tokens)} 个需求关键词"
    if matched:
        reason += f"：{', '.join(matched[:8])}"
    return score, reason + "。"


def _deployment_score(project: dict[str, Any]) -> tuple[int, str]:
    fields = [
        "supports_local_deploy",
        "has_dockerfile",
        "has_docker_compose",
        "has_docker",
        "supports_docker",
        "supports_local_model",
        "supports_ollama",
    ]
    score = round(min(20, _boolean_score(project, fields, 20)))
    return score, f"本地部署、容器与本地模型相关能力命中 {sum(bool(project.get(f)) for f in fields)} 项。"


def _developer_score(project: dict[str, Any]) -> tuple[int, str]:
    fields = [
        "developer_friendly",
        "has_tests",
        "has_ci",
        "has_examples",
        "has_docs",
        "has_pyproject",
        "has_requirements",
        "has_package_json",
    ]
    base = _boolean_score(project, fields, 18)
    structure = str(project.get("project_structure_quality", "")).lower()
    structure_bonus = 2 if structure in {"良好", "good", "excellent"} else 0
    score = round(min(20, base + structure_bonus))
    return score, f"测试、CI、示例、依赖管理和结构质量综合得分 {score}/20。"


def _community_score(project: dict[str, Any], now: datetime) -> tuple[int, str]:
    stars = project.get("stars", 0)
    try:
        stars_value = max(0, int(stars))
    except (TypeError, ValueError):
        stars_value = 0
    star_score = min(9.0, math.log10(stars_value + 1) / 5 * 9)

    updated_at = project.get("updated_at")
    recency_score = 0.0
    if isinstance(updated_at, str) and updated_at.strip():
        try:
            updated = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            age_days = max(0, (now - updated.astimezone(timezone.utc)).days)
            if age_days <= 90:
                recency_score = 6
            elif age_days <= 365:
                recency_score = 5
            elif age_days <= 730:
                recency_score = 3
            else:
                recency_score = 1
        except ValueError:
            recency_score = 0

    score = round(min(15, star_score + recency_score))
    return score, f"Stars={stars_value}，最近更新时间={updated_at or '未知'}。"


def _documentation_score(project: dict[str, Any]) -> tuple[int, str]:
    has_readme = bool(
        project.get("has_readme")
        or project.get("readme_summary")
        or project.get("purpose")
    )
    has_docs = bool(project.get("has_docs"))
    evidence = project.get("evidence")
    evidence_count = len(evidence) if isinstance(evidence, list) else 0
    score = (5 if has_readme else 0) + (3 if has_docs else 0) + min(2, evidence_count)
    return score, f"README={'有' if has_readme else '无'}，文档目录={'有' if has_docs else '无'}，证据 {evidence_count} 条。"


def _license_score(project: dict[str, Any]) -> tuple[int, str]:
    license_name = str(project.get("license") or "").strip()
    if not license_name:
        return 0, "未提供许可证信息。"
    normalised = license_name.upper()
    if normalised in FRIENDLY_LICENSES:
        return 5, f"许可证 {license_name} 较宽松。"
    if "GPL" in normalised or "AGPL" in normalised:
        return 2, f"许可证 {license_name} 有较强传播或网络分发约束。"
    return 3, f"许可证 {license_name} 需要进一步人工确认。"


def score_projects(
    projects: list[dict[str, Any]] | str,
    requirements: dict[str, Any] | str | None = None,
    project_metadata: list[dict[str, Any]] | str | None = None,
) -> str:
    """
    Score projects with the fixed 30/20/20/15/10/5 rubric.

    Inputs may be Python values or JSON strings. The returned value is a JSON
    array sorted by total_score descending.
    """
    requirement_data = _as_dict(requirements or {})
    project_data = _merge_project_data(
        _as_list(projects),
        _as_list(project_metadata or []),
    )
    now = datetime.now(timezone.utc)
    results: list[dict[str, Any]] = []

    for project in project_data:
        name = _project_name(project)
        function_score, function_reason = _function_match(project, requirement_data)
        deployment_score, deployment_reason = _deployment_score(project)
        developer_score, developer_reason = _developer_score(project)
        community_score, community_reason = _community_score(project, now)
        documentation_score, documentation_reason = _documentation_score(project)
        license_score, license_reason = _license_score(project)
        scores = {
            "function_match": function_score,
            "deployment": deployment_score,
            "developer_friendliness": developer_score,
            "community_activity": community_score,
            "documentation": documentation_score,
            "license_friendliness": license_score,
        }
        evidence = project.get("evidence")
        results.append(
            {
                "project": name,
                "full_name": name,
                "total_score": sum(scores.values()),
                "scores": scores,
                "reason": " ".join(
                    [
                        function_reason,
                        deployment_reason,
                        developer_reason,
                        community_reason,
                        documentation_reason,
                        license_reason,
                    ]
                ),
                "evidence": evidence if isinstance(evidence, list) else [],
                "weights": dict(SCORE_WEIGHTS),
            }
        )

    results.sort(key=lambda item: (-item["total_score"], item["project"]))
    for rank, item in enumerate(results, 1):
        item["rank"] = rank
    return json.dumps(results, ensure_ascii=False, indent=2)


def validate_project_result(
    projects: list[dict[str, Any]] | str,
    source_projects: list[dict[str, Any]] | str | None = None,
) -> str:
    """
    Validate candidate or scoring output and return JSON quality findings.

    Checks empty candidates, required fields, README/license presence, score
    ranges, total-score arithmetic, ordering, and suspicious unsupported fields.
    """
    items = _as_list(projects)
    sources = {
        _project_name(item): item
        for item in _as_list(source_projects or [])
        if isinstance(item, dict) and _project_name(item)
    }
    errors: list[str] = []
    warnings: list[str] = []
    known_score_fields = set(SCORE_WEIGHTS)

    if not items:
        errors.append("候选项目为空。")

    previous_total: float | None = None
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict):
            errors.append(f"第 {index} 项不是 JSON 对象。")
            continue
        name = _project_name(item) or f"第 {index} 项"
        if not _project_name(item):
            errors.append(f"{name} 缺少 project/full_name/name。")

        source = sources.get(_project_name(item), item)
        has_readme = bool(
            source.get("has_readme")
            or source.get("readme_summary")
            or source.get("purpose")
        )
        if not has_readme:
            warnings.append(f"{name} 缺少 README 或 README 摘要。")
        if not source.get("license"):
            warnings.append(f"{name} 缺少 license。")

        if "scores" in item:
            scores = item.get("scores")
            if not isinstance(scores, dict):
                errors.append(f"{name} 的 scores 不是对象。")
                continue
            missing = known_score_fields - set(scores)
            unknown = set(scores) - known_score_fields
            if missing:
                errors.append(f"{name} 缺少评分维度：{sorted(missing)}。")
            if unknown:
                warnings.append(f"{name} 出现未定义评分维度：{sorted(unknown)}。")
            for dimension, maximum in SCORE_WEIGHTS.items():
                value = scores.get(dimension)
                if not isinstance(value, (int, float)):
                    errors.append(f"{name} 的 {dimension} 不是数值。")
                elif not 0 <= value <= maximum:
                    errors.append(f"{name} 的 {dimension}={value} 超出 0-{maximum}。")

            total = item.get("total_score")
            if not isinstance(total, (int, float)):
                errors.append(f"{name} 缺少数值 total_score。")
            else:
                calculated = sum(
                    value
                    for key, value in scores.items()
                    if key in known_score_fields and isinstance(value, (int, float))
                )
                if abs(total - calculated) > 0.01:
                    errors.append(f"{name} 的 total_score={total}，但分项合计为 {calculated}。")
                if not 0 <= total <= 100:
                    errors.append(f"{name} 的 total_score={total} 超出 0-100。")
                if previous_total is not None and total > previous_total:
                    errors.append("评分结果未按 total_score 降序排列。")
                previous_total = total

            for field in ("reason", "evidence"):
                if field not in item:
                    errors.append(f"{name} 缺少 {field}。")

        suspicious = {"official", "production_ready", "market_share", "benchmark_winner"}
        unsupported = sorted(suspicious.intersection(item) - set(source))
        if unsupported:
            warnings.append(f"{name} 出现疑似无来源字段：{unsupported}。")

    return json.dumps(
        {"ok": not errors, "errors": errors, "warnings": warnings},
        ensure_ascii=False,
        indent=2,
    )


def validate_report(report: str, known_projects: list[dict[str, Any]] | str | None = None) -> str:
    """
    Validate the final Markdown report structure and source references.

    Requires all nine report sections, at least one Markdown table, and explicit
    references. It also flags project names that are not present upstream.
    """
    errors: list[str] = []
    warnings: list[str] = []
    text = str(report or "")

    if not text.strip():
        errors.append("报告为空。")
    for section in REPORT_SECTIONS:
        if not re.search(rf"^##+\s*(?:\d+\.\s*)?{re.escape(section)}\s*$", text, re.MULTILINE):
            errors.append(f"Markdown 缺少章节：{section}。")
    if not re.search(r"^\|.+\|$", text, re.MULTILINE):
        errors.append("报告缺少 Markdown 对比表格。")

    source_section = re.search(
        r"^##+\s*(?:9\.\s*)?参考来源\s*$([\s\S]*)",
        text,
        re.MULTILINE,
    )
    if not source_section or not re.search(r"https?://|README(?:\.md)?|GitHub", source_section.group(1)):
        errors.append("报告未在参考来源章节提供 README、GitHub 或 URL 来源。")

    known_names = {
        _project_name(item)
        for item in _as_list(known_projects or [])
        if isinstance(item, dict) and _project_name(item)
    }
    github_names = set(re.findall(r"\b[\w.-]+/[\w.-]+\b", text))
    unknown_names = sorted(github_names - known_names) if known_names else []
    if unknown_names:
        warnings.append(f"报告出现上游候选中不存在的项目名：{unknown_names}。")

    return json.dumps(
        {"ok": not errors, "errors": errors, "warnings": warnings},
        ensure_ascii=False,
        indent=2,
    )
