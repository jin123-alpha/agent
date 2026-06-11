from dataclasses import dataclass
from typing import Callable


@dataclass
class HandoffDecision:
    target_agent: str | None = None
    reason: str = ""

    @property
    def should_handoff(self) -> bool:
        return self.target_agent is not None


@dataclass
class Handoff:
    """
    Placeholder for multi-agent collaboration routing.
    """

    name: str
    can_handle: Callable[[str], bool]
    target_agent: str

    def decide(self, user_input: str) -> HandoffDecision:
        if self.can_handle(user_input):
            return HandoffDecision(
                target_agent=self.target_agent,
                reason=f"Matched handoff rule: {self.name}",
            )

        return HandoffDecision()
