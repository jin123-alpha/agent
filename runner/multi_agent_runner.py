"""
多 Agent 流水线 Runner。

按 AGENT_PIPELINE 顺序执行 Agent，每个 Agent 的输出通过 HandoffDecision.context_data
传递给下一个 Agent，最终返回包含所有阶段结果的 RunResult。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent import Agent
from agent.agents import AGENT_PIPELINE, create_all_agents
from result import RunResult, StreamEvent
from tracing import Tracer

from .config import load_config
from .runner import build_initial_state, check_guardrails, create_agent_graph

ToolCallback = Callable[[str], None]


def _inject_context(user_input: str, context_data: dict[str, Any]) -> str:
    """将前序 Agent 输出的 context_data 注入到下一个 Agent 的 prompt 中。"""
    if not context_data:
        return user_input

    context_json = json.dumps(context_data, ensure_ascii=False, indent=2)
    return (
        f"## 前序 Agent 传递的上下文数据\n"
        f"```json\n{context_json}\n```\n\n"
        f"## 原始用户需求\n{user_input}"
    )


def _extract_output(final_messages: list) -> str:
    """从 LangGraph 最终状态中提取 Agent 的文本输出。"""
    for msg in reversed(final_messages):
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def run_multi_agent_pipeline(
    user_input: str,
    max_tool_calls: int = 10,
    config: dict | None = None,
    agents: dict[str, Agent] | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
    on_agent_start: Callable[[str], None] | None = None,
    on_agent_end: Callable[[str, str], None] | None = None,
) -> RunResult:
    """
    按流水线顺序执行 5 个 Agent。

    Parameters
    ----------
    user_input : str
        用户的原始需求文本。
    agents : dict[str, Agent] | None
        {name: Agent} 字典，默认自动创建。
    on_agent_start : callback(agent_name)
        Agent 开始执行时回调。
    on_agent_end : callback(agent_name, output)
        Agent 执行结束时回调。

    Returns
    -------
    RunResult
        final_output 为最后一个 Agent（ReportAgent）的输出，
        metadata 包含每个 Agent 的输出和 handoff 决策链。
    """
    config = config or load_config()
    agents = agents or create_all_agents()

    context_data: dict[str, Any] = {}
    pipeline_trace: list[dict[str, Any]] = []
    tracer = Tracer()

    for agent_name in AGENT_PIPELINE:
        agent = agents[agent_name]

        # --- Guardrail 检查 ---
        blocked = check_guardrails(agent, user_input)
        if blocked:
            pipeline_trace.append(
                {"agent": agent_name, "status": "blocked", "reason": blocked}
            )
            return RunResult(
                final_output=blocked,
                metadata={
                    "blocked_by_guardrail": True,
                    "blocked_agent": agent_name,
                    "pipeline_trace": pipeline_trace,
                },
            )

        # --- 注入上下文 ---
        prompt = _inject_context(user_input, context_data)

        if on_agent_start:
            on_agent_start(agent_name)

        tracer.record("pipeline_agent_start", agent_name=agent_name)

        # --- 执行 Agent ---
        graph, HumanMessage = create_agent_graph(
            config,
            max_tool_calls,
            agent=agent,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        initial_state = build_initial_state(HumanMessage, prompt)
        final_state = graph.invoke(initial_state)

        agent_output = _extract_output(final_state["messages"])

        tracer.record("pipeline_agent_end", agent_name=agent_name)

        if on_agent_end:
            on_agent_end(agent_name, agent_output)

        # --- Handoff：将当前 Agent 输出存入 context_data ---
        context_data[agent_name] = agent_output

        # --- 记录 handoff 决策 ---
        handoff_decision = None
        for handoff in agent.handoffs:
            decision = handoff.decide(user_input, context_data=context_data)
            if decision.should_handoff:
                handoff_decision = decision
                break

        pipeline_trace.append(
            {
                "agent": agent_name,
                "status": "completed",
                "output_length": len(agent_output),
                "handoff_to": handoff_decision.target_agent
                if handoff_decision
                else None,
            }
        )

    # --- 返回最终结果 ---
    final_output = context_data.get(AGENT_PIPELINE[-1], "")

    return RunResult(
        final_output=final_output,
        metadata={
            "pipeline_trace": pipeline_trace,
            "agent_outputs": context_data,
        },
    )


def stream_multi_agent_events(
    user_input: str,
    max_tool_calls: int = 10,
    config: dict | None = None,
    agents: dict[str, Agent] | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
):
    """
    流式版本的多 Agent 流水线，yield StreamEvent。
    每个 Agent 切换时 yield 分隔事件。
    """
    config = config or load_config()
    agents = agents or create_all_agents()

    context_data: dict[str, Any] = {}

    for agent_name in AGENT_PIPELINE:
        agent = agents[agent_name]

        yield StreamEvent(
            type="text",
            content=f"\n\n{'='*60}\n🤖 [{agent_name}] 开始执行...\n{'='*60}\n\n",
        )

        blocked = check_guardrails(agent, user_input)
        if blocked:
            yield StreamEvent(type="text", content=f"⛔ 被 Guardrail 阻断: {blocked}")
            return

        prompt = _inject_context(user_input, context_data)

        graph, HumanMessage = create_agent_graph(
            config,
            max_tool_calls,
            agent=agent,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )
        initial_state = build_initial_state(HumanMessage, prompt)
        final_state = graph.invoke(initial_state)

        agent_output = _extract_output(final_state["messages"])
        context_data[agent_name] = agent_output

        yield StreamEvent(type="text", content=agent_output)
        yield StreamEvent(
            type="text",
            content=f"\n✅ [{agent_name}] 完成\n",
        )
