from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class StreamEvent:
    type: Literal["text", "line_break"]
    content: str = ""


@dataclass
class RunResult:
    final_output: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
