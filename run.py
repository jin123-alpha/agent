from result import print_run_result, print_stream_events, tool_progress_callbacks
from runner import (
    load_config,
    run_agent,
    run_multi_agent_pipeline,
    stream_agent_events,
    stream_multi_agent_events,
)


def main(user_input: str) -> None:
    """Run the default single-agent mode."""
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


def pipeline(user_input: str) -> None:
    """Run the multi-agent GitHub technology selection pipeline."""
    config = load_config()
    max_tool_calls = 10

<<<<<<< HEAD
    print(f"\n用户需求：{user_input}\n")
=======
    print(f"\n📋 用户需求: {user_input}\n")
>>>>>>> c8d5269 (调试)
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
                on_agent_start=lambda name: print(f"\n当前 Agent：{name}"),
                on_agent_end=lambda name, _: print(f"Agent 完成：{name}"),
            )
            print_run_result(result)


if __name__ == "__main__":
    demo_input = (
        "我想做一个本地部署的 RAG 知识库系统，要求支持 PDF 上传、"
        "向量检索、Ollama、本地模型、Web UI，并且方便二次开发。"
    )
    pipeline(demo_input)

    # main(demo_input)
