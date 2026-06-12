import json
import unittest

from guardrail.output_guardrails import (
    REPORT_OUTPUT_GUARDRAIL,
    SCORING_OUTPUT_GUARDRAIL,
)
from result import RunResult
from tools.scoring_tools import (
    REPORT_SECTIONS,
    SCORE_WEIGHTS,
    score_projects,
    validate_project_result,
    validate_report,
)


class ScoringQualityTests(unittest.TestCase):
    def setUp(self):
        self.requirements = {
            "keywords": ["RAG", "Ollama", "Web UI"],
            "constraints": ["local deployment"],
        }
        self.projects = [
            {
                "full_name": "example/local-rag",
                "description": "Local deployment RAG with Ollama and Web UI",
                "readme_summary": "RAG, Ollama, Web UI and Docker",
                "features": ["RAG", "Ollama", "Web UI"],
                "has_readme": True,
                "has_docs": True,
                "has_tests": True,
                "has_ci": True,
                "has_docker": True,
                "supports_local_deploy": True,
                "supports_local_model": True,
                "license": "MIT",
                "stars": 1500,
                "updated_at": "2026-05-01T00:00:00Z",
                "evidence": [
                    {
                        "source": "README.md",
                        "quote": "Run with Docker and Ollama",
                        "feature": "supports_local_deploy",
                    }
                ],
            }
        ]

    def test_score_projects_uses_required_100_point_rubric(self):
        scores = json.loads(score_projects(self.projects, self.requirements))
        self.assertEqual(len(scores), 1)
        self.assertEqual(scores[0]["weights"], SCORE_WEIGHTS)
        self.assertEqual(scores[0]["total_score"], sum(scores[0]["scores"].values()))
        self.assertLessEqual(scores[0]["total_score"], 100)
        self.assertEqual(scores[0]["rank"], 1)

    def test_project_validator_detects_missing_readme_and_license(self):
        result = json.loads(validate_project_result([{"full_name": "x/y"}]))
        self.assertTrue(any("README" in item for item in result["warnings"]))
        self.assertTrue(any("license" in item for item in result["warnings"]))

    def test_scoring_guardrail_recalculates_total(self):
        scores = json.loads(score_projects(self.projects, self.requirements))
        scores[0]["total_score"] += 1
        result = SCORING_OUTPUT_GUARDRAIL.run(
            json.dumps(scores, ensure_ascii=False),
            {"RepoAnalysisAgent": self.projects},
        )
        self.assertFalse(result.ok)
        self.assertTrue(any("分项合计" in item for item in result.failures))

    def test_report_validator_requires_all_sections_and_sources(self):
        report = "# 技术选型报告\n\n"
        for index, section in enumerate(REPORT_SECTIONS, 1):
            report += f"## {index}. {section}\n"
            report += (
                "| 项目 | 分数 |\n|---|---|\n| example/local-rag | 80 |\n"
                if index == 2
                else "内容。\n"
            )
        report += "\n- GitHub: https://github.com/example/local-rag\n- README.md\n"
        validation = json.loads(validate_report(report, self.projects))
        self.assertTrue(validation["ok"], validation)
        self.assertTrue(
            REPORT_OUTPUT_GUARDRAIL.run(
                report,
                {"GitHubSearchAgent": self.projects},
            ).ok
        )

    def test_run_result_has_stable_metadata_schema(self):
        result = RunResult(final_output="ok")
        self.assertEqual(result.metadata["requirements"], {})
        self.assertEqual(result.metadata["projects"], [])
        self.assertEqual(result.metadata["scores"], [])
        self.assertEqual(result.metadata["guardrail_warnings"], [])
        self.assertEqual(result.metadata["agent_steps"], [])


if __name__ == "__main__":
    unittest.main()
