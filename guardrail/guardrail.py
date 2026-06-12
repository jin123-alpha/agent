from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GuardrailResult:
    """输入 Guardrail 检查结果。"""

    ok: bool
    message: str = ""


@dataclass
class Guardrail:
    """
    输入侧 Guardrail —— Agent 执行前对 user_input 做安全/合规检查。
    """

    name: str
    check: Callable[[str], GuardrailResult]

    def run(self, value: str) -> GuardrailResult:
        return self.check(value)


# ---------------------------------------------------------------------------
# Output Guardrail —— Agent 执行后对输出做纯规则校验
# ---------------------------------------------------------------------------


@dataclass
class OutputGuardrailResult:
    """输出 Guardrail 检查结果，支持多条失败详情。"""

    ok: bool
    failures: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.ok:
            return ""
        return "; ".join(self.failures)


@dataclass
class OutputGuardrail:
    """
    输出侧 Guardrail —— Agent 执行后对其输出做纯规则校验。

    与 Guardrail 的区别：
    - 检查的是 Agent 输出（而非用户输入）
    - check 函数接收 (output, context_data) 两个参数
    - 支持返回多条失败详情
    - 检查失败会触发重试（而非直接阻断）

    Parameters
    ----------
    name : str
        Guardrail 名称
    check : Callable
        (output: str, context: dict) -> OutputGuardrailResult
    """

    name: str
    check: Callable[[str, dict[str, Any]], OutputGuardrailResult]

    def run(self, output: str, context: dict[str, Any] | None = None) -> OutputGuardrailResult:
        return self.check(output, context or {})
