import unittest

from agent.critic_agent import (
    STAGE_CHECKLISTS,
    build_critic_instructions,
    create_critic_agent,
    get_checklist,
)


class CriticAgentTests(unittest.TestCase):
    def test_repo_analysis_checklist_covers_coverage_and_evidence(self):
        checklist = get_checklist("RepoAnalysisAgent")
        self.assertEqual(checklist.stage, "RepoAnalysisAgent")

        item_names = [item.name for item in checklist.items]
        self.assertIn("证据字段", item_names)
        self.assertIn("结果覆盖与唯一性", item_names)
        self.assertIn("字段一致性", item_names)
        self.assertIn("结论与证据闭环", item_names)

    def test_repo_analysis_prompt_includes_stage_specific_guidance(self):
        prompt = build_critic_instructions(STAGE_CHECKLISTS["RepoAnalysisAgent"])
        self.assertIn("RepoAnalysisAgent 专属审查重点", prompt)
        self.assertIn("full_name 是否保持一致", prompt)
        self.assertIn("dependency_files", prompt)
        self.assertIn("project_structure_quality", prompt)

    def test_repo_analysis_critic_agent_metadata_is_stage_bound(self):
        critic = create_critic_agent("RepoAnalysisAgent")
        self.assertEqual(critic.metadata["critic_stage"], "RepoAnalysisAgent")
        self.assertEqual(critic.metadata["max_retries"], 2)


if __name__ == "__main__":
    unittest.main()