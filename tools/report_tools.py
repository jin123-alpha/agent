from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Any


REPORT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports"


def _json_value(value: Any, default: Any) -> Any:
    if isinstance(value, type(default)):
        return value
    if not isinstance(value, str):
        return default

    text = value.strip()
    if not text:
        return default
    if "```" in text:
        import re

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return default
    return parsed if isinstance(parsed, type(default)) else default


def _as_list(value: Any) -> list[dict[str, Any]]:
    parsed = _json_value(value, [])
    return [item for item in parsed if isinstance(item, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return _json_value(value, {})


def _project_name(item: dict[str, Any]) -> str:
    return str(
        item.get("project")
        or item.get("full_name")
        or item.get("name")
        or ""
    ).strip()


def _by_name(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {_project_name(item): item for item in items if _project_name(item)}


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        rows = [["No data" for _ in headers]]
    header = "| " + " | ".join(headers) + " |"
    divider = "| " + " | ".join("---" for _ in headers) + " |"
    body = [
        "| " + " | ".join(str(value).replace("\n", " ").strip() for value in row) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _join_values(value: Any) -> str:
    if isinstance(value, list):
        joined = ", ".join(str(item) for item in value if item)
        return joined or "Not confirmed"
    return str(value or "Not confirmed")


def generate_report(
    requirements: dict[str, Any] | str,
    projects: list[dict[str, Any]] | str,
    analysed_projects: list[dict[str, Any]] | str | None = None,
    scores: list[dict[str, Any]] | str | None = None,
    guardrail_warnings: list[str] | str | None = None,
) -> str:
    """
    Generate a deterministic Markdown technical selection report.

    Inputs may be Python objects or JSON strings. The output always uses the
    required nine-section report structure.
    """
    requirement_data = _as_dict(requirements)
    project_data = _as_list(projects)
    analysis_data = _as_list(analysed_projects or [])
    score_data = _as_list(scores or [])
    warnings = _json_value(guardrail_warnings or [], [])
    if not isinstance(warnings, list):
        warnings = []

    projects_by_name = _by_name(project_data)
    analysis_by_name = _by_name(analysis_data)
    score_by_name = _by_name(score_data)

    ordered_names = [_project_name(item) for item in score_data if _project_name(item)]
    for item in project_data + analysis_data:
        name = _project_name(item)
        if name and name not in ordered_names:
            ordered_names.append(name)

    overview_rows = []
    github_rows = []
    feature_rows = []
    score_rows = []
    source_lines = []

    for name in ordered_names:
        project = projects_by_name.get(name, {})
        analysis = analysis_by_name.get(name, {})
        score = score_by_name.get(name, {})
        merged = {**project, **analysis, **score}

        overview_rows.append(
            [
                score.get("rank", ""),
                name,
                project.get("description")
                or analysis.get("readme_summary")
                or "Not confirmed",
            ]
        )
        github_rows.append(
            [
                name,
                project.get("stars", "Not confirmed"),
                project.get("language", "Not confirmed"),
                project.get("license") or analysis.get("license") or "Not confirmed",
                project.get("updated_at", "Not confirmed"),
            ]
        )
        feature_rows.append(
            [
                name,
                _join_values(analysis.get("features") or analysis.get("main_features")),
                "Yes" if merged.get("has_docker") or merged.get("supports_docker") else "Not confirmed",
                "Yes" if merged.get("has_docs") else "Not confirmed",
                analysis.get("project_structure_quality", "Not confirmed"),
            ]
        )

        scores_obj = score.get("scores") if isinstance(score.get("scores"), dict) else {}
        score_rows.append(
            [
                score.get("rank", ""),
                name,
                scores_obj.get("function_match", ""),
                scores_obj.get("deployment", ""),
                scores_obj.get("developer_friendliness", ""),
                scores_obj.get("community_activity", ""),
                scores_obj.get("documentation", ""),
                scores_obj.get("license_friendliness", ""),
                score.get("total_score", ""),
            ]
        )

        url = project.get("html_url")
        if url:
            source_lines.append(f"- {name}: {url}")
        if analysis.get("readme_summary") or analysis.get("evidence"):
            source_lines.append(f"- {name}: README.md")

    best = score_data[0] if score_data else {}
    best_name = _project_name(best) or (ordered_names[0] if ordered_names else "No recommendation")
    warning_lines = [f"- {item}" for item in warnings] or ["- No automated guardrail warnings."]
    source_lines = source_lines or ["- No upstream source URL was available."]

    requirement_lines = [
        f"- Project type: {requirement_data.get('project_type', 'Not specified')}",
        f"- Keywords: {_join_values(requirement_data.get('keywords'))}",
        f"- Language: {requirement_data.get('language', 'Not specified')}",
        f"- Constraints: {_join_values(requirement_data.get('constraints'))}",
        f"- Preferences: {_join_values(requirement_data.get('preferences'))}",
    ]

    return "\n\n".join(
        [
            "# 技术选型报告",
            "## 1. 用户需求分析\n" + "\n".join(requirement_lines),
            "## 2. 候选项目概览\n" + _markdown_table(["Rank", "Project", "Summary"], overview_rows),
            "## 3. GitHub 基础信息对比\n"
            + _markdown_table(["Project", "Stars", "Language", "License", "Updated At"], github_rows),
            "## 4. 功能特性对比\n"
            + _markdown_table(["Project", "Features", "Docker", "Docs", "Structure"], feature_rows),
            "## 5. 项目评分矩阵\n"
            + _markdown_table(
                [
                    "Rank",
                    "Project",
                    "Function",
                    "Deploy",
                    "Developer",
                    "Community",
                    "Docs",
                    "License",
                    "Total",
                ],
                score_rows,
            ),
            f"## 6. 推荐方案\n推荐项目：{best_name}。\n\n"
            f"推荐依据：{best.get('reason', 'No scoring reason was provided.')}",
            "## 7. 风险分析\n" + "\n".join(warning_lines),
            "## 8. 后续开发路线建议\n"
            "1. Clone and run the top-ranked project locally.\n"
            "2. Verify PDF ingestion, vector retrieval, local model integration, and Web UI workflows.\n"
            "3. Build a minimal business-specific prototype on top of the selected project.\n"
            "4. Add tests, deployment scripts, and operational documentation before production use.",
            "## 9. 参考来源\n" + "\n".join(dict.fromkeys(source_lines)),
        ]
    )


def save_markdown_report(
    markdown: str,
    run_id: str,
    output_dir: str | Path | None = None,
    project_type: str | None = None,
) -> str:
    report_dir = Path(output_dir).expanduser().resolve() if output_dir else REPORT_DIR
    report_dir.mkdir(parents=True, exist_ok=True)

    name = unicodedata.normalize("NFKC", str(project_type or "")).strip()
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', " ", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    if not name or name.upper() in {
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    }:
        name = "技术选型报告"
    name = name[:80].rstrip(" .")

    path = report_dir / f"{name}.md"
    suffix = 2
    while path.exists():
        path = report_dir / f"{name}_{suffix}.md"
        suffix += 1

    path.write_text(markdown, encoding="utf-8")
    return str(path.resolve())
