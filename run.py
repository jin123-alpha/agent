from result import print_run_result, print_stream_events, tool_progress_callbacks
from runner import load_config, run_agent, stream_agent_events


def main() -> None:
    config = load_config()
    user_input = "上海嘉定今天天气如何"
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


if __name__ == "__main__":
    main()
