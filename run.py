import sys

from result import print_run_result, print_stream_events, tool_progress_callbacks
from runner import (
    load_config,
    run_agent,
    run_multi_agent_pipeline,
    stream_agent_events,
    stream_multi_agent_events,
)


def main(user_input :str) -> None:
    """默认单 Agent 模式。"""
    config = load_config()

    max_tool_calls = 10

    with tool_progress_callbacks() as callbacks:
        if config["stream"]:
            events = stream_agent_events(
                user_input,
                max_tool_calls=max_tool_calls,
                config=config,
                on_tool_start=callbacks["on_tool_start"],
                on_tool_end=callbacks["on_tool_end"],
            )
            print_stream_events(events)
        else:
            result = run_agent(
                user_input,
                max_tool_calls=max_tool_calls,
                config=config,
                on_tool_start=callbacks["on_tool_start"],
                on_tool_end=callbacks["on_tool_end"],
            )
            print_run_result(result)


def pipeline(user_input:str) -> None:
    """多 Agent 流水线模式 —— GitHub 开源技术选型。"""
    config = load_config()
    max_tool_calls = 10

    print(f"\n📋 用户需求: {user_input}\n")

    with tool_progress_callbacks() as callbacks:
        if config.get("stream", False):
            events = stream_multi_agent_events(
                user_input,
                max_tool_calls=max_tool_calls,
                config=config,
                on_tool_start=callbacks["on_tool_start"],
                on_tool_end=callbacks["on_tool_end"],
            )
            print_stream_events(events)
        else:
            result = run_multi_agent_pipeline(
                user_input,
                max_tool_calls=max_tool_calls,
                config=config,
                on_tool_start=callbacks["on_tool_start"],
                on_tool_end=callbacks["on_tool_end"],
                on_agent_start=lambda name: print(f"\n🤖 [{name}] 开始执行..."),
                on_agent_end=lambda name, _: print(f"✅ [{name}] 完成"),
            )
            print_run_result(result)


if __name__ == "__main__":
    user_input = "帮我找一个agent构建相关的仓库，要求是Python写的，star数超过1000，最近一年有更新的。"
    pipeline(user_input)
    
    #main(user_input)

