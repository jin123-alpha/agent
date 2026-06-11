from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class TraceEvent:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


@dataclass
class Tracer:
    """
    Collects lightweight debug events for a run.
    """

    events: list[TraceEvent] = field(default_factory=list)

    def record(self, name: str, **data) -> None:
        self.events.append(TraceEvent(name=name, data=data))
