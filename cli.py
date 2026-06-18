from __future__ import annotations

import argparse
import io
import json
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from result import StreamEvent
from runner import load_config, stream_multi_agent_events


APP_TITLE = "GitHub 技术选型助手"
ACCENT = "dark_orange"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one GitHub technology-selection task."
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Single task prompt. If omitted, the CLI asks interactively.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to a JSON configuration file. Defaults to ./config.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / "reports",
        help="Report directory. Defaults to ./reports in the launch directory.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=10,
        help="Maximum tool-call rounds for each Agent.",
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="Also save Trace and LangGraph state snapshots.",
    )
    return parser


def _short(value: Any, limit: int = 72) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, list):
        text = "、".join(str(item) for item in value)
    elif isinstance(value, dict):
        text = "、".join(f"{key}: {item}" for key, item in value.items())
    else:
        text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _parse_output(content: str) -> Any | None:
    text = content.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _boolean_mark(value: Any) -> str:
    if value is True:
        return "[green]✓[/green]"
    if value is False:
        return "[red]–[/red]"
    return "[dim]?[/dim]"


def _base_table(*columns: tuple[str, dict[str, Any]]) -> Table:
    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold cyan",
        padding=(0, 1),
        expand=True,
    )
    for name, options in columns:
        table.add_column(name, **options)
    return table


def _render_requirement(data: dict[str, Any]) -> Table:
    table = _base_table(
        ("需求维度", {"style": "bold", "width": 14}),
        ("解析结果", {"ratio": 1}),
    )
    rows = [
        ("项目类型", data.get("project_type")),
        ("技术关键词", data.get("keywords")),
        ("开发语言", data.get("language")),
        ("硬性约束", data.get("constraints")),
        ("偏好条件", data.get("preferences")),
        ("候选数量", data.get("top_n")),
    ]
    for label, value in rows:
        table.add_row(label, _short(value))
    return table


def _render_search(data: list[dict[str, Any]]) -> Table:
    table = _base_table(
        ("#", {"justify": "right", "width": 3}),
        ("候选仓库", {"ratio": 2}),
        ("Stars", {"justify": "right", "width": 9}),
        ("语言", {"width": 10}),
        ("最近更新", {"width": 12}),
    )
    for index, repo in enumerate(data, 1):
        updated = str(repo.get("updated_at") or "-")[:10]
        stars = repo.get("stars")
        table.add_row(
            str(index),
            _short(repo.get("full_name"), 40),
            f"{stars:,}" if isinstance(stars, int) else _short(stars),
            _short(repo.get("language"), 10),
            updated,
        )
    return table


def _render_analysis(data: list[dict[str, Any]]) -> Table:
    table = _base_table(
        ("仓库", {"ratio": 2}),
        ("结构", {"width": 8}),
        ("测试", {"justify": "center", "width": 6}),
        ("文档", {"justify": "center", "width": 6}),
        ("CI", {"justify": "center", "width": 5}),
        ("Docker", {"justify": "center", "width": 8}),
        ("License", {"width": 10}),
        ("主要特性", {"ratio": 3}),
    )
    for repo in data:
        table.add_row(
            _short(repo.get("full_name"), 32),
            _short(repo.get("project_structure_quality"), 8),
            _boolean_mark(repo.get("has_tests")),
            _boolean_mark(repo.get("has_docs")),
            _boolean_mark(repo.get("has_ci")),
            _boolean_mark(repo.get("has_docker")),
            _short(repo.get("license"), 10),
            _short(repo.get("features"), 62),
        )
    return table


def _render_scoring(data: list[dict[str, Any]]) -> Table:
    dimension_names: list[str] = []
    for item in data:
        scores = item.get("scores")
        if isinstance(scores, dict):
            for name in scores:
                if name not in dimension_names:
                    dimension_names.append(name)

    table = _base_table(
        ("排名", {"justify": "center", "width": 6}),
        ("项目", {"ratio": 2}),
        *[
            (name, {"justify": "right", "width": max(7, min(len(name) + 2, 12))})
            for name in dimension_names
        ],
        ("总分", {"justify": "right", "style": "bold green", "width": 7}),
    )
    for index, item in enumerate(data, 1):
        scores = item.get("scores") if isinstance(item.get("scores"), dict) else {}
        table.add_row(
            str(item.get("rank", index)),
            _short(item.get("project") or item.get("full_name"), 36),
            *[_short(scores.get(name), 8) for name in dimension_names],
            _short(item.get("total_score"), 8),
        )
    return table


def _render_critic(data: dict[str, Any]) -> Group:
    passed = data.get("passed") is True
    status = (
        Text("语义审查通过", style="bold green")
        if passed
        else Text("语义审查未通过", style="bold yellow")
    )
    table = _base_table(
        ("检查项", {"ratio": 2}),
        ("结果", {"justify": "center", "width": 8}),
        ("说明", {"ratio": 3}),
    )
    checks = data.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if not isinstance(check, dict):
                continue
            table.add_row(
                _short(check.get("name"), 42),
                _boolean_mark(check.get("passed")),
                _short(check.get("detail"), 90),
            )
    comment = Text(_short(data.get("overall_comment"), 180), style="dim")
    return Group(status, table, comment)


def _agent_output_panel(event: StreamEvent) -> Panel:
    agent_name = event.agent_name or "Agent"
    data = _parse_output(event.content)
    renderable: Any | None = None

    if agent_name == "RequirementAgent" and isinstance(data, dict):
        renderable = _render_requirement(data)
    elif agent_name == "GitHubSearchAgent" and isinstance(data, list):
        renderable = _render_search(
            [item for item in data if isinstance(item, dict)]
        )
    elif agent_name == "RepoAnalysisAgent" and isinstance(data, list):
        renderable = _render_analysis(
            [item for item in data if isinstance(item, dict)]
        )
    elif agent_name == "ScoringAgent" and isinstance(data, list):
        renderable = _render_scoring(
            [item for item in data if isinstance(item, dict)]
        )
    elif agent_name.startswith("CriticAgent") and isinstance(data, dict):
        renderable = _render_critic(data)
    elif agent_name == "ReportAgent":
        renderable = Text("最终技术选型报告已整理完成。", style="green")

    if renderable is None:
        renderable = Text(
            f"阶段已完成，生成 {_short(len(event.content))} 个字符的结构化结果。",
            style="dim",
        )

    return Panel(
        renderable,
        title=f"[bold]{agent_name}[/bold]",
        title_align="left",
        border_style="cyan",
        padding=(0, 1),
    )


def _render_agent_output(console: Console, event: StreamEvent) -> None:
    console.print(_agent_output_panel(event))


# OSC 8 hyperlink sequences: ESC ] 8 ; params ; URI ST  (ST = ESC \ or BEL).
# Rich emits these for Markdown links when force_terminal=True, but
# prompt_toolkit's ANSI parser can't decode them and leaks the raw
# "8;id=...;URL" / "8;;" payloads into the visible text. Strip the wrappers
# and keep the link text that sits between them.
_OSC8_RE = re.compile(r"\x1b\]8;[^;]*;[^\x1b\x07]*(?:\x1b\\|\x07)")


def _renderable_to_ansi(renderable: Any, width: int = 100) -> str:
    output = io.StringIO()
    render_console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=max(60, width),
    )
    render_console.print(renderable)
    return _OSC8_RE.sub("", output.getvalue()).rstrip()


def _print_header(console: Console) -> None:
    mark = Text()
    mark.append("◆", style=f"bold {ACCENT}")
    mark.append("  ")
    mark.append(APP_TITLE, style="bold white")
    mark.append("\n   多 Agent GitHub 项目检索、分析与技术选型", style="dim")
    console.print(
        Panel(
            mark,
            box=box.SQUARE,
            border_style=ACCENT,
            padding=(1, 2),
        )
    )


def _read_prompt(console: Console) -> str:
    console.print(f"[{ACCENT}]╭─[/] [bold]You[/bold]")
    prompt = console.input(f"[{ACCENT}]│[/]  [bold {ACCENT}]❯[/] ")
    console.print(f"[{ACCENT}]╰─[/]")
    return prompt.strip()


def _print_user_message(console: Console, prompt: str) -> None:
    console.print(
        Panel(
            Text(prompt),
            title="[bold]You[/bold]",
            title_align="left",
            border_style=ACCENT,
            box=box.ROUNDED,
            padding=(0, 1),
        )
    )


def _locked_input() -> Panel:
    return Panel(
        Text("❯  任务运行中，输入已锁定", style="dim"),
        title="[bold]You[/bold]",
        title_align="left",
        border_style="bright_black",
        box=box.ROUNDED,
        padding=(0, 1),
    )


def _run_task(
    prompt: str,
    args: argparse.Namespace,
    console: Console,
    config: dict[str, Any],
) -> int:
    output_dir = args.output_dir.expanduser().resolve()
    live = Live(
        _locked_input(),
        console=console,
        refresh_per_second=10,
        transient=True,
        vertical_overflow="visible",
    )
    live_started = False
    current_agent: str | None = None
    current_activity: str | None = None

    def stop_status() -> None:
        live.update(_locked_input(), refresh=True)

    def start_status(agent_name: str) -> None:
        nonlocal current_agent, current_activity
        current_agent = agent_name
        current_activity = None
        live.update(
            Group(
                Spinner(
                    "dots",
                    text=f"[bold cyan]{agent_name} loading...[/bold cyan]",
                    style=ACCENT,
                ),
                _locked_input(),
            ),
            refresh=True,
        )

    def update_status(tool_name: str | None = None) -> None:
        if current_agent is None:
            return
        suffix = f"  [dim]调用 {tool_name}[/dim]" if tool_name else ""
        live.update(
            Group(
                Spinner(
                    "dots",
                    text=(
                        f"[bold cyan]{current_agent} loading..."
                        f"[/bold cyan]{suffix}"
                    ),
                    style=ACCENT,
                ),
                _locked_input(),
            ),
            refresh=True,
        )

    def update_activity(message: str) -> None:
        nonlocal current_activity
        current_activity = message
        if current_agent is None:
            return
        live.update(
            Group(
                Spinner(
                    "dots",
                    text=(
                        f"[bold cyan]{current_agent} loading...[/bold cyan] "
                        f"[dim]{current_activity}[/dim]"
                    ),
                    style=ACCENT,
                ),
                _locked_input(),
            ),
            refresh=True,
        )

    def on_tool_start(tool_name: str) -> None:
        update_status(tool_name)

    def on_tool_end(_tool_name: str) -> None:
        update_status()

    try:
        live.start(refresh=True)
        live_started = True
        events = stream_multi_agent_events(
            prompt,
            max_tool_calls=args.max_tool_calls,
            config=config,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
            on_status=update_activity,
            report_output_dir=str(output_dir),
            save_debug_artifacts=args.debug_artifacts,
        )

        pending_output: StreamEvent | None = None
        for event in events:
            if pending_output is not None and event.type not in {"retry", "error"}:
                _render_agent_output(console, pending_output)
                pending_output = None

            if event.type == "agent_start":
                start_status(event.agent_name or "Agent")
            elif event.type == "agent_output":
                stop_status()
                pending_output = event
            elif event.type == "retry":
                stop_status()
                pending_output = None
                retry_count = event.data.get("retry_count", "?")
                source = event.data.get("source", "review")
                console.print(
                    f"[yellow]↻ {event.agent_name} 第 {retry_count} 次重试[/yellow] "
                    f"[dim]({source})[/dim]"
                )
                start_status(event.agent_name or "Agent")
            elif event.type == "error":
                stop_status()
                pending_output = None
                live.stop()
                live_started = False
                console.print(
                    Panel(
                        event.content,
                        title=event.agent_name or "运行失败",
                        border_style="red",
                    )
                )
                return 1
            elif event.type == "completed":
                stop_status()
                live.stop()
                live_started = False
                report_path = event.data.get("report_path")
                if event.content.strip():
                    console.print(
                        Panel(
                            Markdown(event.content),
                            title="[bold]Assistant · 技术选型报告[/bold]",
                            title_align="left",
                            border_style="green",
                            box=box.ROUNDED,
                            padding=(1, 2),
                        )
                    )

                lines = ["[bold green]报告已生成并保存[/bold green]"]
                if report_path:
                    lines.append(f"[dim]报告已保存至[/dim]  {report_path}")
                console.print(
                    Panel(
                        "\n".join(lines),
                        title="[bold]Assistant[/bold]",
                        title_align="left",
                        border_style="green",
                        box=box.ROUNDED,
                    )
                )
                if args.debug_artifacts:
                    if event.data.get("trace_path"):
                        console.print(f"[dim]Trace: {event.data['trace_path']}[/dim]")
                    if event.data.get("state_path"):
                        console.print(f"[dim]State: {event.data['state_path']}[/dim]")
                return 0

        if pending_output is not None:
            _render_agent_output(console, pending_output)
    except KeyboardInterrupt:
        if live_started:
            live.stop()
            live_started = False
        console.print("\n[yellow]任务已取消。[/yellow]")
        return 130
    except Exception as exc:
        if live_started:
            live.stop()
            live_started = False
        console.print(
            Panel(
                str(exc),
                title="运行异常",
                border_style="red",
            )
        )
        return 1
    finally:
        if live_started:
            live.stop()

    console.print("[red]流水线未返回完成事件。[/red]")
    return 1


def _run_fullscreen_cli(
    args: argparse.Namespace,
    config: dict[str, Any],
) -> int:
    from prompt_toolkit import ANSI, Application
    from prompt_toolkit.filters import Condition
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import HSplit, Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    from prompt_toolkit.layout.dimension import Dimension
    from prompt_toolkit.mouse_events import MouseEventType
    from prompt_toolkit.styles import Style
    from prompt_toolkit.widgets import Frame, TextArea

    state: dict[str, Any] = {
        "busy": False,
        "agent": "",
        "tool": "",
        "activity": "",
        "spinner": 0,
        "exit_code": 0,
        "follow_bottom": True,
        "notice": "就绪 · 输入新任务，Ctrl+C 退出",
    }

    # History is stored as Rich renderables and rendered to ANSI lines lazily,
    # cached per width. Only the visible slice is handed to the Window, so a
    # frame costs O(visible rows) instead of re-rendering the whole scrollback.
    entries: list[Any] = []
    line_cache: dict[str, Any] = {"width": -1, "lines": [], "dirty": True}
    state["scroll"] = 0
    state["view_height"] = 1

    def _viewport_height() -> int:
        # Total rows minus chrome: header(2) + separator(1) + status(1)
        # + input frame(5). Clamp so we always show something.
        rows = app.output.get_size().rows
        return max(3, rows - 9)

    def _content_width() -> int:
        return max(60, app.output.get_size().columns - 4)

    def _rebuild_lines() -> None:
        width = _content_width()
        lines: list[str] = []
        for index, renderable in enumerate(entries):
            if index:
                lines.append("")
            lines.extend(_renderable_to_ansi(renderable, width).split("\n"))
        line_cache["width"] = width
        line_cache["lines"] = lines
        line_cache["dirty"] = False

    def _ensure_lines() -> list[str]:
        if line_cache["dirty"] or line_cache["width"] != _content_width():
            _rebuild_lines()
        return line_cache["lines"]

    def _max_scroll() -> int:
        return max(0, len(_ensure_lines()) - state["view_height"])

    def _clamp_scroll() -> None:
        if state["follow_bottom"]:
            state["scroll"] = _max_scroll()
        else:
            state["scroll"] = max(0, min(state["scroll"], _max_scroll()))

    def history_text():
        if not entries:
            return FormattedText(
                [("class:muted", "在下方输入技术选型需求，然后按 Enter。")]
            )
        lines = _ensure_lines()
        state["view_height"] = _viewport_height()
        _clamp_scroll()
        top = state["scroll"]
        visible = lines[top:top + state["view_height"]]
        return ANSI("\n".join(visible))

    scroll_holder: dict[str, Any] = {}

    def _scroll_by(delta: int) -> None:
        if delta < 0:
            state["follow_bottom"] = False
        state["scroll"] = max(0, min(state["scroll"] + delta, _max_scroll()))
        if delta > 0 and state["scroll"] >= _max_scroll():
            state["follow_bottom"] = True
        app_instance = scroll_holder.get("app")
        if app_instance is not None:
            app_instance.invalidate()

    class HistoryControl(FormattedTextControl):
        def mouse_handler(self, mouse_event):
            if mouse_event.event_type == MouseEventType.SCROLL_UP:
                _scroll_by(-3)
                return None
            if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
                _scroll_by(3)
                return None
            return super().mouse_handler(mouse_event)

    history_control = HistoryControl(
        text=history_text,
        focusable=True,
        show_cursor=False,
    )
    # wrap_lines is False: lines are pre-wrapped at render width by Rich.
    history_window = Window(
        content=history_control,
        wrap_lines=False,
        right_margins=[],
    )

    status_control = FormattedTextControl(
        text=lambda: (
            [
                (
                    "class:status",
                    f"{'⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏'[state['spinner'] % 10]} "
                    f"{state['agent']} loading..."
                    + (
                        f"  {state['activity']}"
                        if state["activity"]
                        else (f"  调用 {state['tool']}" if state["tool"] else "")
                    ),
                )
            ]
            if state["busy"]
            else [("class:muted", state["notice"])]
        )
    )
    status_window = Window(
        content=status_control,
        height=1,
        style="class:status-line",
    )

    input_area: TextArea

    def append_renderable(renderable: Any) -> None:
        entries.append(renderable)
        if len(entries) > 120:
            del entries[:-120]
        line_cache["dirty"] = True
        if state["follow_bottom"]:
            state["scroll"] = _max_scroll()
        app.invalidate()

    def unlock_input(message: str = "可以继续输入下一个任务") -> None:
        state["busy"] = False
        state["agent"] = ""
        state["tool"] = ""
        state["activity"] = ""
        state["notice"] = message
        input_area.buffer.read_only = Condition(lambda: state["busy"])
        app.layout.focus(input_area)
        app.invalidate()

    def run_pipeline(prompt: str) -> None:
        pending_output: StreamEvent | None = None

        def ui(callback, *values) -> None:
            app.loop.call_soon_threadsafe(callback, *values)

        def set_agent(agent_name: str) -> None:
            state["agent"] = agent_name
            state["tool"] = ""
            state["activity"] = ""
            app.invalidate()

        def set_tool(tool_name: str) -> None:
            state["tool"] = tool_name
            app.invalidate()

        def clear_tool(_tool_name: str) -> None:
            state["tool"] = ""
            app.invalidate()

        def set_activity(message: str) -> None:
            state["activity"] = message
            app.invalidate()

        try:
            events = stream_multi_agent_events(
                prompt,
                max_tool_calls=args.max_tool_calls,
                config=config,
                on_tool_start=lambda name: ui(set_tool, name),
                on_tool_end=lambda name: ui(clear_tool, name),
                on_status=lambda message: ui(set_activity, message),
                report_output_dir=str(args.output_dir.expanduser().resolve()),
                save_debug_artifacts=args.debug_artifacts,
            )

            for event in events:
                if pending_output is not None and event.type not in {"retry", "error"}:
                    ui(append_renderable, _agent_output_panel(pending_output))
                    pending_output = None

                if event.type == "agent_start":
                    ui(set_agent, event.agent_name or "Agent")
                elif event.type == "agent_output":
                    pending_output = event
                elif event.type == "retry":
                    pending_output = None
                    retry_count = event.data.get("retry_count", "?")
                    ui(
                        append_renderable,
                        Text(
                            f"↻ {event.agent_name} 第 {retry_count} 次重试",
                            style="yellow",
                        ),
                    )
                elif event.type == "error":
                    pending_output = None
                    ui(
                        append_renderable,
                        Panel(
                            event.content,
                            title=event.agent_name or "运行失败",
                            border_style="red",
                        ),
                    )
                    ui(unlock_input, "任务失败，可以修改需求后重新提交")
                    return
                elif event.type == "completed":
                    report_path = event.data.get("report_path")
                    if event.content.strip():
                        ui(
                            append_renderable,
                            Panel(
                                Markdown(event.content),
                                title="[bold]Assistant · 技术选型报告[/bold]",
                                title_align="left",
                                border_style="green",
                                box=box.ROUNDED,
                                padding=(1, 2),
                            ),
                        )
                    message = Text("报告已生成并保存", style="bold green")
                    if report_path:
                        message.append(f"\n{report_path}", style="dim")
                    ui(
                        append_renderable,
                        Panel(
                            message,
                            title="[bold]Assistant[/bold]",
                            title_align="left",
                            border_style="green",
                            box=box.ROUNDED,
                        ),
                    )
                    ui(unlock_input, "任务完成，可以继续输入下一个任务")
                    return
        except Exception as exc:
            ui(
                append_renderable,
                Panel(str(exc), title="运行异常", border_style="red"),
            )
            ui(unlock_input)

    def submit(buffer) -> bool:
        prompt = buffer.text.strip()
        if not prompt or state["busy"]:
            return True

        buffer.text = ""
        state["busy"] = True
        state["notice"] = ""
        input_area.buffer.read_only = Condition(lambda: state["busy"])
        append_renderable(
            Panel(
                Text(prompt),
                title="[bold]You[/bold]",
                title_align="left",
                border_style=ACCENT,
                box=box.ROUNDED,
                padding=(0, 1),
            )
        )
        threading.Thread(target=run_pipeline, args=(prompt,), daemon=True).start()
        return True

    input_area = TextArea(
        height=3,
        prompt="❯ ",
        multiline=False,
        wrap_lines=True,
        accept_handler=submit,
        read_only=Condition(lambda: state["busy"]),
    )
    input_frame = Frame(input_area, title="You", style="class:input-frame")

    header = Window(
        content=FormattedTextControl(
            [
                ("class:brand", "◆  GitHub 技术选型助手\n"),
                ("class:muted", "   多 Agent GitHub 项目检索、分析与技术选型"),
            ]
        ),
        height=Dimension.exact(2),
        style="class:header",
    )

    bindings = KeyBindings()

    @bindings.add("c-c")
    def exit_app(event) -> None:
        event.app.exit(result=state["exit_code"])

    @bindings.add("pageup")
    def page_up(_event) -> None:
        _scroll_by(-(state["view_height"] - 1 or 1))

    @bindings.add("pagedown")
    def page_down(_event) -> None:
        _scroll_by(state["view_height"] - 1 or 1)

    @bindings.add("home")
    def scroll_home(_event) -> None:
        state["follow_bottom"] = False
        state["scroll"] = 0
        app.invalidate()

    @bindings.add("end")
    def scroll_end(_event) -> None:
        state["follow_bottom"] = True
        state["scroll"] = _max_scroll()
        app.invalidate()

    @bindings.add("c-up")
    def history_up(_event) -> None:
        _scroll_by(-1)

    @bindings.add("c-down")
    def history_down(_event) -> None:
        _scroll_by(1)

    root = HSplit(
        [
            header,
            Window(height=1, char="─", style="class:separator"),
            history_window,
            status_window,
            input_frame,
        ]
    )
    app: Application[int] = Application(
        layout=Layout(root, focused_element=input_area),
        key_bindings=bindings,
        full_screen=True,
        mouse_support=True,
        style=Style.from_dict(
            {
                "brand": "bold #ff8700",
                "muted": "#808080",
                "header": "bg:#181818",
                "separator": "#444444",
                "status": "bold #00afff",
                "status-line": "bg:#181818",
                "input-frame": "#ff8700",
            }
        ),
    )
    scroll_holder["app"] = app

    def animate_spinner() -> None:
        while True:
            time.sleep(0.12)
            if state["busy"] and app.loop is not None:
                state["spinner"] += 1
                app.loop.call_soon_threadsafe(app.invalidate)

    threading.Thread(target=animate_spinner, daemon=True).start()

    initial_prompt = (args.prompt or "").strip()
    if initial_prompt:
        def start_initial_task() -> None:
            input_area.buffer.text = initial_prompt
            submit(input_area.buffer)

        return app.run(pre_run=start_initial_task)
    return app.run()


def run_cli(args: argparse.Namespace, console: Console | None = None) -> int:
    console = console or Console()
    interactive = console.is_terminal

    if args.max_tool_calls < 1:
        console.print("[red]--max-tool-calls 必须大于 0。[/red]")
        return 2

    config_path = args.config.resolve() if args.config else None
    if config_path and not config_path.is_file():
        console.print(f"[red]配置文件不存在：{config_path}[/red]")
        return 2

    try:
        config = load_config(config_path)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    if interactive and console.file is sys.stdout:
        return _run_fullscreen_cli(args, config)

    if interactive:
        console.clear()
    _print_header(console)

    prompt = (args.prompt or "").strip()
    first_task = True

    while True:
        if not prompt:
            if not interactive:
                console.print("[red]Prompt 不能为空。[/red]")
                return 2
            try:
                prompt = _read_prompt(console)
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]会话已结束。[/dim]")
                return 0
            if not prompt:
                console.print("[yellow]请输入任务，或按 Ctrl+C 退出。[/yellow]")
                continue
        elif first_task:
            _print_user_message(console, prompt)

        exit_code = _run_task(prompt, args, console, config)
        if exit_code != 0 or not interactive:
            return exit_code

        first_task = False
        prompt = ""


def main() -> int:
    return run_cli(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
