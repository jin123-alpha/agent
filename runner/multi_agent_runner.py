"""
多 Agent 流水线 Runner（含 Critic 重试机制）。

按 AGENT_PIPELINE 顺序执行 Agent，每个 Agent 的输出通过 HandoffDecision.context_data
传递给下一个 Agent。当遇到 CriticAgent 时，如果审查不通过且未超过重试上限，
则回退重新执行被审查的 Agent。
"""

from __future__ import annotations

import json
from typing import Any, Callable

from agent import Agent
from agent.agents import AGENT_CRITIC, AGENT_PIPELINE, create_all_agents
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


def _inject_retry_context(
    user_input: str,
    context_data: dict[str, Any],
    critic_feedback: str,
    retry_count: int,
) -> str:
    """注入上下文 + Critic 反馈，用于重试。"""
    base_prompt = _inject_context(user_input, context_data)
    return (
        f"{base_prompt}\n\n"
        f"## ⚠️ Critic 审查未通过（第 {retry_count} 次重试）\n"
        f"请根据以下反馈修正你的输出：\n"
        f"```\n{critic_feedback}\n```\n"
    )


def _extract_output(final_messages: list) -> str:
    """从 LangGraph 最终状态中提取 Agent 的文本输出。"""
    for msg in reversed(final_messages):
        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _is_critic(agent_name: str) -> bool:
    """判断是否为 CriticAgent。"""
    return agent_name.startswith(AGENT_CRITIC)


def _get_reviewed_stage(critic_name: str) -> str:
    """从 CriticAgent 名称中提取被审查的 Agent 名。如 CriticAgent_GitHubSearchAgent → GitHubSearchAgent"""
    return critic_name[len(AGENT_CRITIC) + 1 :]


def _parse_critic_result(critic_output: str) -> dict[str, Any]:
    """解析 Critic 的 JSON 输出。"""
    try:
        # 尝试提取 JSON（可能被 markdown 代码块包裹）
        text = critic_output.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            text = "\n".join(json_lines)
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # 如果解析失败，默认通过
        return {"passed": True, "overall_comment": "Critic 输出解析失败，默认通过"}


def _run_single_agent(
    agent: Agent,
    prompt: str,
    config: dict,
    max_tool_calls: int,
    on_tool_start: ToolCallback | None,
    on_tool_end: ToolCallback | None,
) -> str:
    """执行单个 Agent 并返回文本输出。"""
    graph, HumanMessage = create_agent_graph(
        config,
        max_tool_calls,
        agent=agent,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )
    initial_state = build_initial_state(HumanMessage, prompt)
    final_state = graph.invoke(initial_state)
    return _extract_output(final_state["messages"])


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
    按流水线顺序执行 Agent（含 Critic 重试）。

    当 CriticAgent 审查不通过时，回退重新执行被审查的 Agent，
    将 Critic 反馈注入 prompt 中，直到通过或达到最大重试次数。
    """
    config = config or load_config()
    agents = agents or create_all_agents()

    context_data: dict[str, Any] = {}
    pipeline_trace: list[dict[str, Any]] = []
    tracer = Tracer()

    i = 0
    retry_counts: dict[str, int] = {}  # stage → 已重试次数

    while i < len(AGENT_PIPELINE):
        agent_name = AGENT_PIPELINE[i]
        agent = agents[agent_name]

        # --- Guardrail ---
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

        # --- 构造 prompt ---
        prompt = _inject_context(user_input, context_data)

        if on_agent_start:
            on_agent_start(agent_name)

        tracer.record("pipeline_agent_start", agent_name=agent_name)

        # --- 执行 ---
        agent_output = _run_single_agent(
            agent, prompt, config, max_tool_calls, on_tool_start, on_tool_end
        )

        tracer.record("pipeline_agent_end", agent_name=agent_name)

        if on_agent_end:
            on_agent_end(agent_name, agent_output)

        # --- Critic 重试逻辑 ---
        if _is_critic(agent_name):
            reviewed_stage = _get_reviewed_stage(agent_name)
            critic_result = _parse_critic_result(agent_output)

            if not critic_result.get("passed", True):
                current_retries = retry_counts.get(reviewed_stage, 0)
                max_retries = agent.metadata.get("max_retries", 2)

                if current_retries < max_retries:
                    retry_counts[reviewed_stage] = current_retries + 1

                    # 提取 Critic 的修改建议
                    feedback = critic_result.get(
                        "retry_suggestion",
                        critic_result.get("overall_comment", "请修正输出"),
                    )

                    pipeline_trace.append(
                        {
                            "agent": agent_name,
                            "status": "retry_triggered",
                            "retry_count": retry_counts[reviewed_stage],
                            "feedback": feedback,
                        }
                    )

                    if on_agent_end:
                        on_agent_end(
                            agent_name,
                            f"❌ 审查未通过，触发第 {retry_counts[reviewed_stage]} 次重试",
                        )

                    # 回退：从被审查的 Agent 重新开始
                    # 先注入 Critic 反馈到 context
                    context_data[f"{agent_name}_feedback"] = feedback

                    # 将 prompt 中加入重试反馈
                    retry_prompt = _inject_retry_context(
                        user_input,
                        context_data,
                        feedback,
                        retry_counts[reviewed_stage],
                    )

                    # 找到被审查 Agent 在 pipeline 中的位置并回退
                    reviewed_idx = AGENT_PIPELINE.index(reviewed_stage)

                    if on_agent_start:
                        on_agent_start(f"{reviewed_stage} (重试 #{retry_counts[reviewed_stage]})")

                    # 重新执行被审查的 Agent
                    reviewed_agent = agents[reviewed_stage]
                    retry_output = _run_single_agent(
                        reviewed_agent,
                        retry_prompt,
                        config,
                        max_tool_calls,
                        on_tool_start,
                        on_tool_end,
                    )

                    if on_agent_end:
                        on_agent_end(reviewed_stage, retry_output)

                    # 更新 context
                    context_data[reviewed_stage] = retry_output

                    pipeline_trace.append(
                        {
                            "agent": reviewed_stage,
                            "status": "retried",
                            "retry_count": retry_counts[reviewed_stage],
                            "output_length": len(retry_output),
                        }
                    )

                    # 不前进 i，重新执行 Critic
                    continue
                else:
                    # 达到最大重试次数，记录并继续
                    pipeline_trace.append(
                        {
                            "agent": agent_name,
                            "status": "max_retries_reached",
                            "retry_count": current_retries,
                        }
                    )

                    if on_agent_end:
                        on_agent_end(
                            agent_name,
                            f"⚠️ 达到最大重试次数 ({max_retries})，继续流水线",
                        )
            else:
                pipeline_trace.append(
                    {
                        "agent": agent_name,
                        "status": "passed",
                    }
                )

            # Critic 输出也存入 context（审查记录）
            context_data[agent_name] = agent_output
        else:
            # 普通 Agent
            context_data[agent_name] = agent_output

            # 记录 handoff
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

        i += 1

    # --- 返回最终结果（取 ReportAgent 输出，而非最后的 Critic 输出）---
    from agent.agents import AGENT_REPORT

    final_output = context_data.get(AGENT_REPORT, "")

    return RunResult(
        final_output=final_output,
        metadata={
            "pipeline_trace": pipeline_trace,
            "agent_outputs": context_data,
            "retry_counts": retry_counts,
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
    含 Critic 重试逻辑。
    """
    config = config or load_config()
    agents = agents or create_all_agents()

    context_data: dict[str, Any] = {}
    retry_counts: dict[str, int] = {}

    i = 0
    while i < len(AGENT_PIPELINE):
        agent_name = AGENT_PIPELINE[i]
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

        agent_output = _run_single_agent(
            agent, prompt, config, max_tool_calls, on_tool_start, on_tool_end
        )

        yield StreamEvent(type="text", content=agent_output)

        if _is_critic(agent_name):
            reviewed_stage = _get_reviewed_stage(agent_name)
            critic_result = _parse_critic_result(agent_output)

            if not critic_result.get("passed", True):
                current_retries = retry_counts.get(reviewed_stage, 0)
                max_retries = agent.metadata.get("max_retries", 2)

                if current_retries < max_retries:
                    retry_counts[reviewed_stage] = current_retries + 1
                    feedback = critic_result.get(
                        "retry_suggestion",
                        critic_result.get("overall_comment", "请修正输出"),
                    )
                    context_data[f"{agent_name}_feedback"] = feedback

                    yield StreamEvent(
                        type="text",
                        content=f"\n🔄 审查未通过，触发第 {retry_counts[reviewed_stage]} 次重试...\n",
                    )

                    retry_prompt = _inject_retry_context(
                        user_input,
                        context_data,
                        feedback,
                        retry_counts[reviewed_stage],
                    )

                    reviewed_agent = agents[reviewed_stage]
                    retry_output = _run_single_agent(
                        reviewed_agent,
                        retry_prompt,
                        config,
                        max_tool_calls,
                        on_tool_start,
                        on_tool_end,
                    )
                    context_data[reviewed_stage] = retry_output

                    yield StreamEvent(type="text", content=retry_output)
                    # 不前进 i，重新 Critic
                    continue
                else:
                    yield StreamEvent(
                        type="text",
                        content=f"\n⚠️ 达到最大重试次数 ({max_retries})，继续\n",
                    )

            context_data[agent_name] = agent_output
        else:
            context_data[agent_name] = agent_output

        yield StreamEvent(
            type="text",
            content=f"\n✅ [{agent_name}] 完成\n",
        )

        i += 1
