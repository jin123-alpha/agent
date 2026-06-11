from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class HandoffDecision:
    """Agent 间路由决策，携带上下文数据用于信息传递。"""

    target_agent: str | None = None
    reason: str = ""
    context_data: dict[str, Any] = field(default_factory=dict)

    @property
    def should_handoff(self) -> bool:
        return self.target_agent is not None


@dataclass
class Handoff:
    """
    Multi-agent collaboration routing.
    can_handle 判断是否路由，target_agent 指定目标 Agent。
    """

    name: str
    can_handle: Callable[[str], bool]
    target_agent: str

    def decide(
        self, user_input: str, context_data: dict[str, Any] | None = None
    ) -> HandoffDecision:
        if self.can_handle(user_input):
            return HandoffDecision(
                target_agent=self.target_agent,
                reason=f"Matched handoff rule: {self.name}",
                context_data=context_data or {},
            )

        return HandoffDecision()
