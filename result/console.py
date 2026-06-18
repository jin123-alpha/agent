import sys
import time
import threading
from contextlib import contextmanager
from collections.abc import Iterable

from .result import DebugResult, RunResult, StreamEvent


def print_stream_events(events: Iterable[StreamEvent]) -> None:
    printed_text = False

    for event in events:
        if event.type == "line_break":
            if printed_text:
                print()
        elif event.type == "text":
            print(event.content, end="", flush=True)
            printed_text = True
        elif event.type == "agent_start":
            print(f"\n\n{'='*60}\n🤖 [{event.agent_name}] 开始执行...\n{'='*60}\n")
            printed_text = True
        elif event.type == "agent_output":
            print(event.content)
            printed_text = True
        elif event.type == "tool_start":
            print(f"\n正在调用工具 {event.tool_name}...")
            printed_text = True
        elif event.type == "tool_end":
            print(f"工具完成：{event.tool_name}")
            printed_text = True
        elif event.type == "retry":
            retry_count = event.data.get("retry_count", "?")
            print(f"\n🔄 [{event.agent_name}] 第 {retry_count} 次重试：{event.content}")
            printed_text = True
        elif event.type == "error":
            print(f"\n⛔ [{event.agent_name or 'Pipeline'}] {event.content}")
            printed_text = True
        elif event.type == "completed":
            report_path = event.data.get("report_path")
            print("\n✅ 流水线完成")
            if report_path:
                print(f"Report saved to: {report_path}")
            printed_text = True

    if printed_text:
        print()


def print_run_result(result: RunResult) -> None:
    report_path = result.metadata.get("report_path")
    trace_path = result.metadata.get("trace_path")
    state_path = result.metadata.get("state_path")
    if report_path:
        print(f"Report saved to: {report_path}")
    if trace_path:
        print(f"Trace saved to: {trace_path}")
    if state_path:
        print(f"State saved to: {state_path}")
    if report_path or trace_path or state_path:
        print()
    if result.final_output:
        print(result.final_output)


def print_debug_result(result: DebugResult) -> None:
    print(f"# Debug Result: {result.agent_name}")
    print("\n## Input")
    print(result.input_data)
    print("\n## Output")
    print(result.output)

    if result.output_guardrail_result is not None:
        print("\n## Output Guardrail")
        print(f"ok: {result.output_guardrail_result.ok}")
        if result.output_guardrail_result.failures:
            for failure in result.output_guardrail_result.failures:
                print(f"- {failure}")

    if result.critic_output:
        print("\n## Critic Output")
        print(result.critic_output)


@contextmanager
def tool_progress_callbacks():
    active_loader = {"stop_event": None, "thread": None}

    def on_tool_start(tool_name: str) -> None:
        stop_event = threading.Event()
        active_loader["stop_event"] = stop_event

        sys.stdout.write("\n")
        sys.stdout.flush()

        def animate():
            frames = ["", ".", "..", "..."]
            index = 0

            while not stop_event.is_set():
                frame = frames[index % len(frames)]
                sys.stdout.write(f"\r正在调用工具 {tool_name}{frame}")
                sys.stdout.flush()
                index += 1
                time.sleep(0.35)

        thread = threading.Thread(target=animate, daemon=True)
        active_loader["thread"] = thread
        thread.start()

    def on_tool_end(tool_name: str) -> None:
        stop_event = active_loader.get("stop_event")
        thread = active_loader.get("thread")

        if stop_event:
            stop_event.set()
        if thread:
            thread.join(timeout=1)

        active_loader["stop_event"] = None
        active_loader["thread"] = None

        sys.stdout.write("\n\n")
        sys.stdout.flush()

    try:
        yield {
            "on_tool_start": on_tool_start,
            "on_tool_end": on_tool_end,
        }
    finally:
        if active_loader.get("stop_event"):
            on_tool_end("")
