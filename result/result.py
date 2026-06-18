from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StreamEvent:
    type: Literal[
        "text",
        "line_break",
        "agent_start",
        "agent_output",
        "tool_start",
        "tool_end",
        "retry",
        "error",
        "completed",
    ]
    content: str = ""
    agent_name: str | None = None
    tool_name: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    final_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Keep the multi-agent result contract stable on success and failure."""
        self.metadata.setdefault("requirements", {})
        self.metadata.setdefault("projects", [])
        self.metadata.setdefault("scores", [])
        self.metadata.setdefault("guardrail_warnings", [])
        self.metadata.setdefault("agent_steps", [])


@dataclass
class DebugResult:
    agent_name: str
    input_data: Any
    output: str
    output_guardrail_result: Any | None = None
    critic_output: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
