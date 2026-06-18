import argparse
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from cli import _render_agent_output, _run_fullscreen_cli, run_cli
from agent import Agent
from agent.agents import AGENT_PIPELINE
from result import StreamEvent
from runner.config import load_config
from runner.multi_agent_runner import stream_multi_agent_events
from tools import ToolRegistry
from tools.report_tools import save_markdown_report


class CliPipelineTests(unittest.TestCase):
    def _agents(self):
        return {
            name: Agent(
                name=name,
                instructions=f"Run {name}",
                tools=ToolRegistry(tools=[]),
            )
            for name in AGENT_PIPELINE
        }

    def test_stream_pipeline_emits_structured_events_and_saves_report(self):
        outputs = {
            "RequirementAgent": '{"project_type": "RAG", "keywords": []}',
            "GitHubSearchAgent": "[]",
            "RepoAnalysisAgent": "[]",
            "ScoringAgent": "[]",
            "ReportAgent": "# 技术选型报告\n\nReportAgent original output",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "runner.multi_agent_runner._run_single_agent",
                side_effect=lambda agent, *_args, **_kwargs: outputs[agent.name],
            ):
                events = list(
                    stream_multi_agent_events(
                        "test prompt",
                        config={},
                        agents=self._agents(),
                        report_output_dir=temp_dir,
                    )
                )

            event_types = [event.type for event in events]
            self.assertEqual(event_types.count("agent_start"), len(AGENT_PIPELINE))
            self.assertEqual(event_types.count("agent_output"), len(AGENT_PIPELINE))
            self.assertEqual(event_types[-1], "completed")

            completed = events[-1]
            report_path = Path(completed.data["report_path"])
            self.assertEqual(report_path.parent, Path(temp_dir).resolve())
            self.assertEqual(report_path.name, "RAG.md")
            self.assertEqual(report_path.read_text(encoding="utf-8"), outputs["ReportAgent"])
            self.assertEqual(completed.content, outputs["ReportAgent"])

    def test_stream_pipeline_forwards_runtime_status_callback(self):
        outputs = {
            "RequirementAgent": '{"project_type": "RAG", "keywords": []}',
            "GitHubSearchAgent": "[]",
            "RepoAnalysisAgent": "[]",
            "ScoringAgent": "[]",
            "ReportAgent": "# report",
        }
        statuses = []

        def run_agent(agent, _prompt, config, *_args, **_kwargs):
            if agent.name == "GitHubSearchAgent":
                config["_runtime_status_callback"]("正在进行语义排序")
            return outputs[agent.name]

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "runner.multi_agent_runner._run_single_agent",
                side_effect=run_agent,
            ):
                list(
                    stream_multi_agent_events(
                        "test prompt",
                        config={},
                        agents=self._agents(),
                        on_status=statuses.append,
                        report_output_dir=temp_dir,
                    )
                )

        self.assertEqual(statuses, ["正在进行语义排序"])

    def test_report_uses_safe_project_type_filename_without_overwriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(
                save_markdown_report(
                    "# 报告",
                    "ignored-run-id",
                    temp_dir,
                    '本地 RAG/知识库:*?"',
                )
            )
            second = Path(
                save_markdown_report(
                    "# 第二份报告",
                    "another-run-id",
                    temp_dir,
                    '本地 RAG/知识库:*?"',
                )
            )

            self.assertEqual(first.name, "本地 RAG 知识库.md")
            self.assertEqual(second.name, "本地 RAG 知识库_2.md")
            self.assertEqual(first.read_text(encoding="utf-8"), "# 报告")

    def test_explicit_invalid_config_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "invalid.json"
            config_path.write_text("{invalid", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "无法读取配置文件"):
                load_config(config_path)

    def test_cli_does_not_print_api_keys(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        args = argparse.Namespace(
            prompt="test prompt",
            config=None,
            output_dir=Path("reports"),
            max_tool_calls=10,
            debug_artifacts=False,
        )
        events = [
            StreamEvent(type="agent_start", agent_name="RequirementAgent"),
            StreamEvent(
                type="agent_output",
                agent_name="RequirementAgent",
                content='{"ok": true}',
            ),
            StreamEvent(
                type="completed",
                content="# report",
                data={"report_path": "reports/test.md"},
            ),
        ]

        with patch("cli.load_config", return_value={"api_key": "super-secret"}):
            with patch("cli.stream_multi_agent_events", return_value=iter(events)):
                exit_code = run_cli(args, console=console)

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertNotIn("super-secret", rendered)
        self.assertIn("report", rendered)
        self.assertIn("reports/test.md", rendered)

    def test_agent_output_is_rendered_as_summary_not_raw_json(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        raw_output = (
            '{"project_type":"agent framework","keywords":["multi-agent","tools"],'
            '"language":"Python","constraints":["open source"],'
            '"preferences":["active"],"top_n":5}'
        )

        _render_agent_output(
            console,
            StreamEvent(
                type="agent_output",
                agent_name="RequirementAgent",
                content=raw_output,
            ),
        )

        rendered = output.getvalue()
        self.assertIn("项目类型", rendered)
        self.assertIn("agent framework", rendered)
        self.assertNotIn('"project_type"', rendered)
        self.assertNotIn(raw_output, rendered)

    def test_cli_hides_stage_output_that_is_immediately_retried(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=False, width=120)
        args = argparse.Namespace(
            prompt="test prompt",
            config=None,
            output_dir=Path("reports"),
            max_tool_calls=10,
            debug_artifacts=False,
        )
        events = [
            StreamEvent(type="agent_start", agent_name="GitHubSearchAgent"),
            StreamEvent(
                type="agent_output",
                agent_name="GitHubSearchAgent",
                content="[]",
            ),
            StreamEvent(
                type="retry",
                agent_name="GitHubSearchAgent",
                content="no candidates",
                data={"source": "output_guardrail", "retry_count": 1},
            ),
            StreamEvent(
                type="agent_output",
                agent_name="GitHubSearchAgent",
                content=(
                    '[{"full_name":"owner/repo","stars":100,'
                    '"language":"Python","updated_at":"2026-01-01"}]'
                ),
            ),
            StreamEvent(
                type="completed",
                content="# report",
                data={"report_path": "reports/test.md"},
            ),
        ]

        with patch("cli.load_config", return_value={}):
            with patch("cli.stream_multi_agent_events", return_value=iter(events)):
                exit_code = run_cli(args, console=console)

        rendered = output.getvalue()
        self.assertEqual(exit_code, 0)
        self.assertIn("owner/repo", rendered)
        self.assertEqual(rendered.count("候选仓库"), 1)

    def test_interactive_cli_accepts_another_task_after_completion(self):
        output = io.StringIO()
        console = Console(file=output, force_terminal=True, width=100)
        args = argparse.Namespace(
            prompt=None,
            config=None,
            output_dir=Path("reports"),
            max_tool_calls=10,
            debug_artifacts=False,
        )

        def completed_events(prompt, **_kwargs):
            return iter(
                [
                    StreamEvent(
                        type="completed",
                        content=f"# {prompt}",
                        data={"report_path": f"reports/{prompt}.md"},
                    )
                ]
            )

        with patch("cli.load_config", return_value={}):
            with patch(
                "cli._read_prompt",
                side_effect=["first-task", "second-task", KeyboardInterrupt],
            ):
                with patch(
                    "cli.stream_multi_agent_events",
                    side_effect=completed_events,
                ) as stream:
                    exit_code = run_cli(args, console=console)

        self.assertEqual(exit_code, 0)
        self.assertEqual(stream.call_count, 2)
        self.assertIn("reports/first-task.md", output.getvalue())
        self.assertIn("reports/second-task.md", output.getvalue())

    def test_fullscreen_cli_accepts_multiple_tasks(self):
        import threading
        import time

        from prompt_toolkit.application import create_app_session
        from prompt_toolkit.input.defaults import create_pipe_input
        from prompt_toolkit.output import DummyOutput

        args = argparse.Namespace(
            prompt=None,
            config=None,
            output_dir=Path("reports"),
            max_tool_calls=10,
            debug_artifacts=False,
        )
        prompts = []

        def completed_events(prompt, **_kwargs):
            prompts.append(prompt)
            return iter(
                [
                    StreamEvent(
                        type="completed",
                        content=f"# {prompt}",
                        data={"report_path": f"reports/{prompt}.md"},
                    )
                ]
            )

        with create_pipe_input() as pipe_input:
            with create_app_session(input=pipe_input, output=DummyOutput()):
                def feed_input():
                    time.sleep(0.05)
                    pipe_input.send_text("first\r")
                    time.sleep(0.2)
                    pipe_input.send_text("second\r")
                    time.sleep(0.2)
                    pipe_input.send_bytes(b"\x03")

                threading.Thread(target=feed_input, daemon=True).start()
                with patch(
                    "cli.stream_multi_agent_events",
                    side_effect=completed_events,
                ):
                    exit_code = _run_fullscreen_cli(args, {})

        self.assertEqual(exit_code, 0)
        self.assertEqual(prompts, ["first", "second"])


if __name__ == "__main__":
    unittest.main()
