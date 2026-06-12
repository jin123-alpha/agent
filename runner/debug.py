import json
from collections.abc import Callable
from typing import Any

from agent import Agent
from guardrail import OutputGuardrail, OutputGuardrailResult
from result import DebugResult

from .config import load_config
from .multi_agent_runner import _inject_context, _run_single_agent

MockFactory = Callable[[], Any]
MockOutputFactory = Callable[[str, dict[str, Any]], str]


def mock_json_input(factory: MockFactory) -> str:
    """
    Build a JSON prompt string from a custom mock factory.
    """
    data = factory()
    if isinstance(data, str):
        return data

    return json.dumps(data, ensure_ascii=False, indent=2)


def run_single_agent_debug(
    agent: Agent,
    mock_input_factory: MockFactory,
    config: dict | None = None,
    max_tool_calls: int = 10,
    context_data: dict[str, Any] | None = None,
    output_guardrail: OutputGuardrail | None = None,
    critic_agent: Agent | None = None,
    mock_output_factory: MockOutputFactory | None = None,
    on_tool_start=None,
    on_tool_end=None,
) -> DebugResult:
    """
    Debug a single agent with custom mock JSON input.

    If mock_output_factory is provided, the LLM call is skipped and the mock output
    is checked by optional OutputGuardrail and CriticAgent.
    """
    config = config or load_config()
    context_data = context_data or {}
    input_prompt = mock_json_input(mock_input_factory)
    prompt = _inject_context(input_prompt, context_data)

    if mock_output_factory:
        output = mock_output_factory(prompt, context_data)
    else:
        output = _run_single_agent(
            agent,
            prompt,
            config,
            max_tool_calls,
            on_tool_start,
            on_tool_end,
        )

    guardrail_result = None
    if output_guardrail:
        guardrail_result = output_guardrail.run(output, context_data)

    critic_output = None
    if critic_agent:
        critic_context = {
            **context_data,
            "debug_agent": agent.name,
            "debug_output": output,
            "output_guardrail": _guardrail_to_dict(guardrail_result),
        }
        critic_prompt = _inject_context(
            "请审查 debug_agent 的 debug_output，并输出审查结论。",
            critic_context,
        )
        critic_output = _run_single_agent(
            critic_agent,
            critic_prompt,
            config,
            max_tool_calls,
            on_tool_start,
            on_tool_end,
        )

    return DebugResult(
        agent_name=agent.name,
        input_data=input_prompt,
        output=output,
        output_guardrail_result=guardrail_result,
        critic_output=critic_output,
        metadata={
            "used_mock_output": mock_output_factory is not None,
            "used_output_guardrail": output_guardrail is not None,
            "used_critic_agent": critic_agent is not None,
        },
    )


def _guardrail_to_dict(result: OutputGuardrailResult | None) -> dict[str, Any] | None:
    if result is None:
        return None

    return {
        "ok": result.ok,
        "failures": result.failures,
        "message": result.message,
    }
