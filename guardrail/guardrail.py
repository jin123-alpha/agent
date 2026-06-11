from dataclasses import dataclass
from typing import Callable


@dataclass
class GuardrailResult:
    ok: bool
    message: str = ""


@dataclass
class Guardrail:
    """
    Placeholder for safety and quality checks around inputs, outputs, or tool calls.
    """

    name: str
    check: Callable[[str], GuardrailResult]

    def run(self, value: str) -> GuardrailResult:
        return self.check(value)
