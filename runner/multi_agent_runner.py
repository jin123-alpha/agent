"""
多 Agent 流水线 Runner（含 OutputGuardrail + Critic 分层审查）。

执行顺序：
1. Agent 执行前：input Guardrail（安全拦截）
2. Agent 执行
3. Agent 执行后：OutputGuardrail（纯规则校验，格式/字段/范围）
   - 不通过 → 直接带规则反馈重试，不消耗 CriticAgent 的 LLM token
4. OutputGuardrail 通过后：CriticAgent（LLM 语义审查）
   - 不通过 → 带语义反馈重试
"""

from __future__ import annotations

import json
import sys
import sys
from typing import Any, Callable, final, final

from agent import Agent
from agent.agents import AGENT_CRITIC, AGENT_PIPELINE, create_all_agents
from guardrail import OutputGuardrailResult
from result import RunResult, StreamEvent
from tracing import Tracer

from .config import load_config
from .runner import build_initial_state, check_guardrails, create_agent_graph
from tools.report_tools import generate_report, save_markdown_report

ToolCallback = Callable[[str], None]
MAX_GUARDRAIL_RETRIES = 2
MAX_CRITIC_FORMAT_RETRIES = 2


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
    feedback: str,
    retry_count: int,
    source: str = "审查",
) -> str:
    """注入上下文 + 反馈，用于重试。source 标记反馈来源（OutputGuardrail / Critic）。"""
    base_prompt = _inject_context(user_input, context_data)
    return (
        f"{base_prompt}\n\n"
        f"## ⚠️ {source}未通过（第 {retry_count} 次重试）\n"
        f"请根据以下反馈修正你的输出：\n"
        f"```\n{feedback}\n```\n"
    )


def _extract_output(final_messages: list) -> str:
    """从 LangGraph 最终状态中提取 Agent 的文本输出。"""
    for msg in reversed(final_messages):
        if getattr(msg, "type", None) == "tool" or msg.__class__.__name__ == "ToolMessage":
            continue

        content = getattr(msg, "content", None)
        if content and isinstance(content, str) and content.strip():
            return content.strip()
    return ""


def _is_critic(agent_name: str) -> bool:
    """判断是否为 CriticAgent。"""
    return agent_name.startswith(AGENT_CRITIC)


def _get_reviewed_stage(critic_name: str) -> str:
    """从 CriticAgent 名称提取被审查 Agent 名。"""
    return critic_name[len(AGENT_CRITIC) + 1 :]


def _rollback_context_from_stage(
    context_data: dict[str, Any],
    stage_name: str,
    preserve_keys: set[str] | None = None,
) -> None:
    """清理被回退阶段及下游上下文，保留明确指定的错误反馈。"""
    preserve_keys = preserve_keys or set()
    if stage_name not in AGENT_PIPELINE:
        return

    stage_index = AGENT_PIPELINE.index(stage_name)
    affected_names = AGENT_PIPELINE[stage_index:]
    keys_to_remove = set(affected_names)

    for name in affected_names:
        keys_to_remove.add(f"{name}_feedback")
        keys_to_remove.add(f"{name}_format_feedback")

    for key in keys_to_remove - preserve_keys:
        context_data.pop(key, None)


def _is_critic_format_error(critic_result: dict[str, Any]) -> bool:
    return critic_result.get("error_type") == "critic_format"


def _parse_critic_result(critic_output: str) -> dict[str, Any]:
    """解析 Critic 的 JSON 输出。"""
    try:
        text = critic_output.strip()
        if "```" in text:
            import re

            fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if fenced_match:
                text = fenced_match.group(1).strip()

        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return {
                "passed": False,
                "error_type": "critic_format",
                "overall_comment": "Critic 输出不是 JSON 对象，不能视为通过",
                "retry_suggestion": "请只输出 Critic 审查 JSON，并包含 passed 字段。",
            }

        if not isinstance(parsed.get("passed"), bool):
            return {
                "passed": False,
                "error_type": "critic_format",
                "overall_comment": "Critic 输出缺少布尔 passed 字段，不能视为通过",
                "retry_suggestion": "请根据审查清单输出包含 passed、checks、evidence_issues、overall_comment、retry_suggestion 的严格 JSON。",
                "raw_output": parsed,
            }

        parsed.setdefault("error_type", "semantic")
        return parsed
    except (json.JSONDecodeError, ValueError):
        return {
            "passed": False,
            "error_type": "critic_format",
            "overall_comment": "Critic 输出解析失败，不能视为通过",
            "retry_suggestion": "请只输出严格 JSON，不要输出工具结果、Markdown 说明或其他文本。",
        }


def _parse_json_output(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return value if isinstance(value, type(default)) else default
    text = value.strip()
    if "```" in text:
        import re

        fenced = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
        if fenced:
            text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, type(default)) else default
    except (json.JSONDecodeError, TypeError):
        return default


def _result_metadata(
    context_data: dict[str, Any],
    pipeline_trace: list[dict[str, Any]],
    **extra: Any,
) -> dict[str, Any]:
    from agent.agents import (
        AGENT_GITHUB_SEARCH,
        AGENT_REQUIREMENT,
        AGENT_SCORING,
    )
    from tools.scoring_tools import validate_project_result

    requirements = _parse_json_output(context_data.get(AGENT_REQUIREMENT, ""), {})
    projects = _parse_json_output(context_data.get(AGENT_GITHUB_SEARCH, ""), [])
    analysed_projects = _parse_json_output(
        context_data.get("RepoAnalysisAgent", ""),
        [],
    )
    scores = _parse_json_output(context_data.get(AGENT_SCORING, ""), [])
    warnings: list[str] = []

    if analysed_projects or projects:
        validation = _parse_json_output(
            validate_project_result(analysed_projects or projects),
            {},
        )
        warnings.extend(validation.get("warnings", []))
        warnings.extend(validation.get("errors", []))
    if scores:
        score_validation = _parse_json_output(
            validate_project_result(scores, analysed_projects),
            {},
        )
        warnings.extend(score_validation.get("warnings", []))
        warnings.extend(score_validation.get("errors", []))

    for entry in pipeline_trace:
        if "failures" in entry:
            warnings.extend(str(item) for item in entry["failures"])
        elif entry.get("status", "").endswith("max_retries_reached"):
            warnings.append(
                f"{entry.get('agent', 'unknown')} reached maximum quality-check retries."
            )

    metadata = {
        "requirements": requirements,
        "projects": projects,
        "scores": scores,
        "guardrail_warnings": list(dict.fromkeys(warnings)),
        "agent_steps": [entry.get("agent") for entry in pipeline_trace if entry.get("agent")],
        "pipeline_trace": pipeline_trace,
        "agent_outputs": context_data,
    }
    metadata.update(extra)
    return metadata


def _build_markdown_report(context_data: dict[str, Any], metadata: dict[str, Any]) -> str:
    analysed_projects = _parse_json_output(
        context_data.get("RepoAnalysisAgent", ""),
        [],
    )
    return generate_report(
        requirements=metadata.get("requirements", {}),
        projects=metadata.get("projects", []),
        analysed_projects=analysed_projects,
        scores=metadata.get("scores", []),
        guardrail_warnings=metadata.get("guardrail_warnings", []),
    )


def _finalize_result(result: RunResult, tracer: Tracer, save_report: bool = False) -> RunResult:
    if save_report and result.final_output.strip():
        result.metadata["report_path"] = save_markdown_report(
            result.final_output,
            tracer.run_id,
        )
    result.metadata["run_id"] = tracer.run_id
    result.metadata["trace_path"] = tracer.save()
    result.metadata["state_path"] = tracer.save_states()
    return result


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
    tracer = agent.tracer
    final_state = dict(initial_state)
    if tracer is not None:
        tracer.record_state(agent.name, "START", final_state)

    for update in graph.stream(initial_state, stream_mode="updates"):
        if not isinstance(update, dict):
            continue
        for step_name, step_update in update.items():
            if not isinstance(step_update, dict):
                continue
            _merge_state_update(final_state, step_update)
            if tracer is not None:
                tracer.record_state(agent.name, step_name, final_state)

    return _extract_output(final_state["messages"])


def _merge_state_update(state: dict[str, Any], update: dict[str, Any]) -> None:
    for key, value in update.items():
        if key == "messages":
            existing = state.setdefault("messages", [])
            if isinstance(value, list):
                existing.extend(value)
            else:
                existing.append(value)
        else:
            state[key] = value


def _run_output_guardrails(
    agent: Agent,
    output: str,
    context_data: dict[str, Any],
) -> OutputGuardrailResult:
    """执行 Agent 的所有 output_guardrails，汇总失败结果。"""
    all_failures: list[str] = []
    for guardrail in agent.output_guardrails:
        result = guardrail.run(output, context_data)
        if not result.ok:
            all_failures.extend(result.failures)
    if all_failures:
        return OutputGuardrailResult(ok=False, failures=all_failures)
    return OutputGuardrailResult(ok=True)


def _build_guardrail_block_result(
    agent_name: str,
    guardrail_result: OutputGuardrailResult,
    retry_count: int,
    pipeline_trace: list[dict[str, Any]],
    context_data: dict[str, Any],
    retry_counts: dict[str, int],
    guardrail_retry_counts: dict[str, int],
) -> RunResult:
    message = (
        f"OutputGuardrail 未通过，且达到最大重试次数 ({MAX_GUARDRAIL_RETRIES})，"
        "流水线停止。"
    )
    pipeline_trace.append({
        "agent": agent_name,
        "status": "guardrail_max_retries_reached",
        "retry_count": retry_count,
        "failures": guardrail_result.failures,
    })
    return RunResult(
        final_output=message,
        metadata=_result_metadata(
            context_data,
            pipeline_trace,
            blocked_by_output_guardrail=True,
            blocked_agent=agent_name,
            failures=guardrail_result.failures,
            retry_counts=retry_counts,
            guardrail_retry_counts=guardrail_retry_counts,
        ),
    )


def _validate_agent_output_with_guardrails(
    agent_name: str,
    agent: Agent,
    agent_output: str,
    user_input: str,
    context_data: dict[str, Any],
    config: dict,
    max_tool_calls: int,
    on_tool_start: ToolCallback | None,
    on_tool_end: ToolCallback | None,
    on_agent_start: Callable[[str], None] | None,
    on_agent_end: Callable[[str, str], None] | None,
    pipeline_trace: list[dict[str, Any]],
    retry_counts: dict[str, int],
    guardrail_retry_counts: dict[str, int],
    tracer: Tracer | None = None,
) -> tuple[str, RunResult | None]:
    """Validate one agent output. Retry locally; block if rules still fail."""
    while True:
        guardrail_result = _run_output_guardrails(agent, agent_output, context_data)
        if guardrail_result.ok:
            if agent.output_guardrails:
                pipeline_trace.append({
                    "agent": agent_name,
                    "status": "guardrail_passed",
                })
            return agent_output, None

        gr_retries = guardrail_retry_counts.get(agent_name, 0)
        if tracer is not None:
            tracer.record(
                "guardrail_warning",
                agent_name=agent_name,
                failures=guardrail_result.failures,
                retry_count=gr_retries,
            )

        if gr_retries >= MAX_GUARDRAIL_RETRIES:
            if on_agent_end:
                on_agent_end(
                    agent_name,
                    f"⚠️ OutputGuardrail 达到最大重试次数 ({MAX_GUARDRAIL_RETRIES})，流水线停止",
                )
            return (
                agent_output,
                _build_guardrail_block_result(
                    agent_name,
                    guardrail_result,
                    gr_retries,
                    pipeline_trace,
                    context_data,
                    retry_counts,
                    guardrail_retry_counts,
                ),
            )

        guardrail_retry_counts[agent_name] = gr_retries + 1
        feedback = guardrail_result.message

        pipeline_trace.append({
            "agent": agent_name,
            "status": "guardrail_retry_triggered",
            "retry_count": guardrail_retry_counts[agent_name],
            "failures": guardrail_result.failures,
        })

        if on_agent_end:
            on_agent_end(
                agent_name,
                f"🛡️ OutputGuardrail 未通过: {feedback}",
            )

        retry_prompt = _inject_retry_context(
            user_input,
            context_data,
            feedback,
            guardrail_retry_counts[agent_name],
            source="OutputGuardrail 规则校验",
        )

        if on_agent_start:
            on_agent_start(f"{agent_name} (规则重试 #{guardrail_retry_counts[agent_name]})")

        agent_output = _run_single_agent(
            agent,
            retry_prompt,
            config,
            max_tool_calls,
            on_tool_start,
            on_tool_end,
        )

        if on_agent_end:
            on_agent_end(agent_name, agent_output)


# ---------------------------------------------------------------------------
# 阻塞式流水线
# ---------------------------------------------------------------------------


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
    分层审查流水线：

    普通 Agent 执行后：
      1. OutputGuardrail（规则层）→ 失败直接重试（不走 CriticAgent）
      2. 通过后继续到 CriticAgent（语义层）→ 失败带反馈重试
    """
    config = config or load_config()
    agents = agents or create_all_agents()

    context_data: dict[str, Any] = {}
    pipeline_trace: list[dict[str, Any]] = []
    tracer = Tracer()
    for agent in agents.values():
        agent.tracer = tracer

    def traced_tool_start(tool_name: str) -> None:
        tracer.record("tool_start", tool_name=tool_name)
        if on_tool_start:
            on_tool_start(tool_name)

    def traced_tool_end(tool_name: str) -> None:
        tracer.record("tool_end", tool_name=tool_name)
        if on_tool_end:
            on_tool_end(tool_name)

    i = 0
    retry_counts: dict[str, int] = {}
    # 区分规则层重试和语义层重试
    guardrail_retry_counts: dict[str, int] = {}
    critic_format_retry_counts: dict[str, int] = {}

    while i < len(AGENT_PIPELINE):
        agent_name = AGENT_PIPELINE[i]
        agent = agents[agent_name]
        # --- 输入 Guardrail ---
        blocked = check_guardrails(agent, user_input)
        if blocked:
            tracer.record("guardrail_warning", agent_name=agent_name, reason=blocked)
            pipeline_trace.append(
                {"agent": agent_name, "status": "blocked", "reason": blocked}
            )
            return _finalize_result(RunResult(
                final_output=blocked,
                metadata=_result_metadata(
                    context_data,
                    pipeline_trace,
                    blocked_by_guardrail=True,
                    blocked_agent=agent_name,
                ),
            ), tracer)

        # --- 构造 prompt ---
        prompt = _inject_context(user_input, context_data)

        if on_agent_start:
            on_agent_start(agent_name)

        tracer.record("pipeline_agent_start", agent_name=agent_name)
  
        # if agent_name == AGENT_PIPELINE[3]:
        #     print(prompt)
        #     sys.exit(0)
  
        # if agent_name == AGENT_PIPELINE[3]:
        #     print(prompt)
        #     sys.exit(0)

        # --- 执行 Agent ---
        agent_output = _run_single_agent(
            agent, prompt, config, max_tool_calls, traced_tool_start, traced_tool_end
        )

        if agent_name == AGENT_PIPELINE[2]:
            print(agent_output)
            sys.exit(0)

        tracer.record("pipeline_agent_end", agent_name=agent_name)
        tracer.record("agent_end", agent_name=agent_name, output_length=len(agent_output))

        if on_agent_end:
            on_agent_end(agent_name, agent_output)

        # ============================================================
        # 分层审查逻辑
        # ============================================================

        if _is_critic(agent_name):
            # ----- CriticAgent（语义层审查） -----
            reviewed_stage = _get_reviewed_stage(agent_name)
            critic_result = _parse_critic_result(agent_output)

            while _is_critic_format_error(critic_result):
                current_format_retries = critic_format_retry_counts.get(agent_name, 0)
                if current_format_retries >= MAX_CRITIC_FORMAT_RETRIES:
                    message = (
                        f"Critic 输出格式无效，且达到最大重试次数 "
                        f"({MAX_CRITIC_FORMAT_RETRIES})，流水线停止。"
                    )
                    pipeline_trace.append({
                        "agent": agent_name,
                        "status": "critic_format_max_retries_reached",
                        "retry_count": current_format_retries,
                        "feedback": critic_result.get("overall_comment", ""),
                    })
                    if on_agent_end:
                        on_agent_end(agent_name, f"⚠️ {message}")
                    return _finalize_result(RunResult(
                        final_output=message,
                        metadata=_result_metadata(
                            context_data,
                            pipeline_trace,
                            blocked_by_critic_format=True,
                            blocked_agent=agent_name,
                            reviewed_stage=reviewed_stage,
                            retry_counts=retry_counts,
                            guardrail_retry_counts=guardrail_retry_counts,
                            critic_format_retry_counts=critic_format_retry_counts,
                        ),
                    ), tracer)

                critic_format_retry_counts[agent_name] = current_format_retries + 1
                feedback = critic_result.get(
                    "retry_suggestion",
                    critic_result.get("overall_comment", "请输出严格 Critic 审查 JSON"),
                )
                context_data[f"{agent_name}_format_feedback"] = feedback

                pipeline_trace.append({
                    "agent": agent_name,
                    "status": "critic_format_retry_triggered",
                    "retry_count": critic_format_retry_counts[agent_name],
                    "feedback": feedback,
                })

                if on_agent_end:
                    on_agent_end(
                        agent_name,
                        f"🧪 Critic 输出格式无效，触发第 {critic_format_retry_counts[agent_name]} 次 Critic 重试",
                    )

                retry_prompt = _inject_retry_context(
                    user_input,
                    context_data,
                    feedback,
                    critic_format_retry_counts[agent_name],
                    source="Critic 输出格式校验",
                )

                if on_agent_start:
                    on_agent_start(f"{agent_name} (格式重试 #{critic_format_retry_counts[agent_name]})")

                agent_output = _run_single_agent(
                    agent,
                    retry_prompt,
                    config,
                    max_tool_calls,
                    traced_tool_start,
                    traced_tool_end,
                )

                if on_agent_end:
                    on_agent_end(agent_name, agent_output)

                critic_result = _parse_critic_result(agent_output)

            context_data.pop(f"{agent_name}_format_feedback", None)

            if not critic_result.get("passed", True):
                current_retries = retry_counts.get(reviewed_stage, 0)
                max_retries = agent.metadata.get("max_retries", 2)

                if current_retries < max_retries:
                    retry_counts[reviewed_stage] = current_retries + 1
                    feedback = critic_result.get(
                        "retry_suggestion",
                        critic_result.get("overall_comment", "请修正输出"),
                    )

                    pipeline_trace.append({
                        "agent": agent_name,
                        "status": "critic_retry_triggered",
                        "retry_count": retry_counts[reviewed_stage],
                        "feedback": feedback,
                    })

                    if on_agent_end:
                        on_agent_end(
                            agent_name,
                            f"❌ Critic 审查未通过，触发第 {retry_counts[reviewed_stage]} 次重试",
                        )

                    feedback_key = f"{agent_name}_feedback"
                    _rollback_context_from_stage(context_data, reviewed_stage)
                    context_data[feedback_key] = feedback

                    retry_prompt = _inject_retry_context(
                        user_input, context_data, feedback,
                        retry_counts[reviewed_stage],
                        source="Critic 语义审查",
                    )

                    if on_agent_start:
                        on_agent_start(f"{reviewed_stage} (Critic 重试 #{retry_counts[reviewed_stage]})")

                    reviewed_agent = agents[reviewed_stage]
                    retry_output = _run_single_agent(
                        reviewed_agent, retry_prompt, config,
                        max_tool_calls, traced_tool_start, traced_tool_end,
                    )

                    if on_agent_end:
                        on_agent_end(reviewed_stage, retry_output)

                    # Critic 触发的重试仍然必须重新经过被审查 Agent 的规则层校验。
                    guardrail_retry_counts.pop(reviewed_stage, None)
                    retry_output, blocked_result = _validate_agent_output_with_guardrails(
                        reviewed_stage,
                        reviewed_agent,
                        retry_output,
                        user_input,
                        context_data,
                        config,
                        max_tool_calls,
                        traced_tool_start,
                        traced_tool_end,
                        on_agent_start,
                        on_agent_end,
                        pipeline_trace,
                        retry_counts,
                        guardrail_retry_counts,
                        tracer,
                    )
                    if blocked_result is not None:
                        return _finalize_result(blocked_result, tracer)

                    context_data[reviewed_stage] = retry_output

                    pipeline_trace.append({
                        "agent": reviewed_stage,
                        "status": "retried_by_critic",
                        "retry_count": retry_counts[reviewed_stage],
                        "output_length": len(retry_output),
                    })

                    # 不前进 i，重新执行 Critic
                    continue
                else:
                    message = f"Critic 审查未通过，且达到最大重试次数 ({max_retries})，流水线停止。"
                    pipeline_trace.append({
                        "agent": agent_name,
                        "status": "critic_max_retries_reached",
                        "retry_count": current_retries,
                    })
                    if on_agent_end:
                        on_agent_end(
                            agent_name,
                            f"⚠️ {message}",
                        )
                    return _finalize_result(RunResult(
                        final_output=message,
                        metadata=_result_metadata(
                            context_data,
                            pipeline_trace,
                            blocked_by_critic=True,
                            blocked_agent=agent_name,
                            reviewed_stage=reviewed_stage,
                            retry_counts=retry_counts,
                            guardrail_retry_counts=guardrail_retry_counts,
                            critic_format_retry_counts=critic_format_retry_counts,
                        ),
                    ), tracer)
            else:
                context_data.pop(f"{agent_name}_feedback", None)
                pipeline_trace.append({
                    "agent": agent_name,
                    "status": "critic_passed",
                })

            context_data[agent_name] = agent_output

        else:
            # ----- 普通 Agent → 先跑 OutputGuardrail（规则层） -----
            agent_output, blocked_result = _validate_agent_output_with_guardrails(
                agent_name,
                agent,
                agent_output,
                user_input,
                context_data,
                config,
                max_tool_calls,
                traced_tool_start,
                traced_tool_end,
                on_agent_start,
                on_agent_end,
                pipeline_trace,
                retry_counts,
                guardrail_retry_counts,
                tracer,
            )
            if blocked_result is not None:
                return _finalize_result(blocked_result, tracer)

            context_data[agent_name] = agent_output

            # 记录 handoff
            handoff_decision = None
            for handoff in agent.handoffs:
                decision = handoff.decide(user_input, context_data=context_data)
                if decision.should_handoff:
                    handoff_decision = decision
                    break

            pipeline_trace.append({
                "agent": agent_name,
                "status": "completed",
                "output_length": len(agent_output),
                "handoff_to": handoff_decision.target_agent if handoff_decision else None,
            })

        i += 1

    # --- 返回最终结果 ---
    from agent.agents import AGENT_REPORT

    metadata = _result_metadata(
        context_data,
        pipeline_trace,
        retry_counts=retry_counts,
        guardrail_retry_counts=guardrail_retry_counts,
        critic_format_retry_counts=critic_format_retry_counts,
    )
    final_output = _build_markdown_report(context_data, metadata)
    context_data[AGENT_REPORT] = final_output
    metadata["agent_outputs"] = context_data

    return _finalize_result(
        RunResult(final_output=final_output, metadata=metadata),
        tracer,
        save_report=True,
    )


# ---------------------------------------------------------------------------
# 流式版本
# ---------------------------------------------------------------------------


def stream_multi_agent_events(
    user_input: str,
    max_tool_calls: int = 10,
    config: dict | None = None,
    agents: dict[str, Agent] | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
):
    """流式流水线，含 OutputGuardrail + Critic 分层重试。"""
    config = config or load_config()
    agents = agents or create_all_agents()

    context_data: dict[str, Any] = {}
    retry_counts: dict[str, int] = {}
    guardrail_retry_counts: dict[str, int] = {}
    critic_format_retry_counts: dict[str, int] = {}

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

            while _is_critic_format_error(critic_result):
                current_format_retries = critic_format_retry_counts.get(agent_name, 0)
                if current_format_retries >= MAX_CRITIC_FORMAT_RETRIES:
                    yield StreamEvent(
                        type="text",
                        content=(
                            f"\n⚠️ Critic 输出格式无效，且达到最大重试次数 "
                            f"({MAX_CRITIC_FORMAT_RETRIES})，流水线停止。\n"
                        ),
                    )
                    return

                critic_format_retry_counts[agent_name] = current_format_retries + 1
                feedback = critic_result.get(
                    "retry_suggestion",
                    critic_result.get("overall_comment", "请输出严格 Critic 审查 JSON"),
                )
                context_data[f"{agent_name}_format_feedback"] = feedback

                yield StreamEvent(
                    type="text",
                    content=(
                        f"\n🧪 Critic 输出格式无效，"
                        f"触发第 {critic_format_retry_counts[agent_name]} 次 Critic 重试...\n"
                    ),
                )

                retry_prompt = _inject_retry_context(
                    user_input,
                    context_data,
                    feedback,
                    critic_format_retry_counts[agent_name],
                    source="Critic 输出格式校验",
                )

                agent_output = _run_single_agent(
                    agent,
                    retry_prompt,
                    config,
                    max_tool_calls,
                    on_tool_start,
                    on_tool_end,
                )

                yield StreamEvent(type="text", content=agent_output)
                critic_result = _parse_critic_result(agent_output)

            context_data.pop(f"{agent_name}_format_feedback", None)

            if not critic_result.get("passed", True):
                current_retries = retry_counts.get(reviewed_stage, 0)
                max_retries = agent.metadata.get("max_retries", 2)

                if current_retries < max_retries:
                    retry_counts[reviewed_stage] = current_retries + 1
                    feedback = critic_result.get(
                        "retry_suggestion",
                        critic_result.get("overall_comment", "请修正输出"),
                    )
                    feedback_key = f"{agent_name}_feedback"
                    _rollback_context_from_stage(context_data, reviewed_stage)
                    context_data[feedback_key] = feedback

                    yield StreamEvent(
                        type="text",
                        content=f"\n🔄 Critic 审查未通过，触发第 {retry_counts[reviewed_stage]} 次重试...\n",
                    )

                    retry_prompt = _inject_retry_context(
                        user_input, context_data, feedback,
                        retry_counts[reviewed_stage],
                        source="Critic 语义审查",
                    )

                    reviewed_agent = agents[reviewed_stage]
                    retry_output = _run_single_agent(
                        reviewed_agent, retry_prompt, config,
                        max_tool_calls, on_tool_start, on_tool_end,
                    )
                    yield StreamEvent(type="text", content=retry_output)

                    guardrail_retry_counts.pop(reviewed_stage, None)
                    while True:
                        guardrail_result = _run_output_guardrails(
                            reviewed_agent,
                            retry_output,
                            context_data,
                        )
                        if guardrail_result.ok:
                            break

                        gr_retries = guardrail_retry_counts.get(reviewed_stage, 0)
                        if gr_retries >= MAX_GUARDRAIL_RETRIES:
                            yield StreamEvent(
                                type="text",
                                content=(
                                    f"\n⚠️ OutputGuardrail 未通过，且达到最大重试次数 "
                                    f"({MAX_GUARDRAIL_RETRIES})，流水线停止。\n"
                                ),
                            )
                            return

                        guardrail_retry_counts[reviewed_stage] = gr_retries + 1
                        yield StreamEvent(
                            type="text",
                            content=(
                                f"\n🛡️ Critic 重试输出未通过 OutputGuardrail: "
                                f"{guardrail_result.message}\n"
                                f"🔄 触发规则层第 {guardrail_retry_counts[reviewed_stage]} 次重试...\n"
                            ),
                        )

                        retry_prompt = _inject_retry_context(
                            user_input,
                            context_data,
                            guardrail_result.message,
                            guardrail_retry_counts[reviewed_stage],
                            source="OutputGuardrail 规则校验",
                        )

                        retry_output = _run_single_agent(
                            reviewed_agent,
                            retry_prompt,
                            config,
                            max_tool_calls,
                            on_tool_start,
                            on_tool_end,
                        )

                        yield StreamEvent(type="text", content=retry_output)

                    context_data[reviewed_stage] = retry_output

                    continue
                else:
                    yield StreamEvent(
                        type="text",
                        content=f"\n⚠️ Critic 审查未通过，且达到最大重试次数 ({max_retries})，流水线停止。\n",
                    )
                    return

            else:
                context_data.pop(f"{agent_name}_feedback", None)

            context_data[agent_name] = agent_output
        else:
            # OutputGuardrail 检查
            while True:
                guardrail_result = _run_output_guardrails(agent, agent_output, context_data)
                if guardrail_result.ok:
                    break

                gr_retries = guardrail_retry_counts.get(agent_name, 0)

                if gr_retries >= MAX_GUARDRAIL_RETRIES:
                    yield StreamEvent(
                        type="text",
                        content=f"\n⚠️ OutputGuardrail 达到最大重试次数，流水线停止\n",
                    )
                    return

                guardrail_retry_counts[agent_name] = gr_retries + 1

                yield StreamEvent(
                    type="text",
                    content=(
                        f"\n🛡️ OutputGuardrail 未通过: {guardrail_result.message}\n"
                        f"🔄 触发规则层第 {guardrail_retry_counts[agent_name]} 次重试...\n"
                    ),
                )

                retry_prompt = _inject_retry_context(
                    user_input, context_data, guardrail_result.message,
                    guardrail_retry_counts[agent_name],
                    source="OutputGuardrail 规则校验",
                )

                agent_output = _run_single_agent(
                    agent, retry_prompt, config,
                    max_tool_calls, on_tool_start, on_tool_end,
                )

                yield StreamEvent(type="text", content=agent_output)

            context_data[agent_name] = agent_output

        yield StreamEvent(
            type="text",
            content=f"\n✅ [{agent_name}] 完成\n",
        )

        i += 1
