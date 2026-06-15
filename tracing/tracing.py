from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


TRACE_DIR = Path(__file__).resolve().parents[1] / "data" / "traces"
STATE_DIR = Path(__file__).resolve().parents[1] / "data" / "states"
MAX_STATE_TEXT_CHARS = 4000


@dataclass
class TraceEvent:
    name: str
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": self.data,
            "timestamp": self.timestamp,
        }


@dataclass
class Tracer:
    """
    Collects lightweight debug events for a run.
    """

    run_id: str = field(default_factory=lambda: uuid4().hex)
    events: list[TraceEvent] = field(default_factory=list)
    state_snapshots: list[dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, **data) -> None:
        self.events.append(TraceEvent(name=name, data=data))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "events": [event.to_dict() for event in self.events],
        }

    def record_state(
        self,
        agent_name: str,
        step_name: str,
        state: dict[str, Any],
    ) -> None:
        self.state_snapshots.append(
            {
                "agent": agent_name,
                "step": step_name,
                "timestamp": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "state": _serialise_state(state),
            }
        )

    def save(self, directory: str | Path | None = None) -> str:
        trace_dir = Path(directory) if directory is not None else TRACE_DIR
        trace_dir.mkdir(parents=True, exist_ok=True)
        path = trace_dir / f"{self.run_id}.json"
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(path)

    def save_states(self, directory: str | Path | None = None) -> str:
        state_dir = Path(directory) if directory is not None else STATE_DIR
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"{self.run_id}.json"
        path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "states": self.state_snapshots,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return str(path)


def _short_text(value: Any) -> str:
    text = str(value or "")
    if len(text) > MAX_STATE_TEXT_CHARS:
        return text[:MAX_STATE_TEXT_CHARS] + "..."
    return text


def _serialise_message(message: Any) -> dict[str, Any]:
    content = getattr(message, "content", "")
    tool_calls = getattr(message, "tool_calls", None)
    result = {
        "class": message.__class__.__name__,
        "type": getattr(message, "type", None),
        "content": _short_text(content),
    }
    if tool_calls:
        result["tool_calls"] = tool_calls
    name = getattr(message, "name", None)
    if name:
        result["name"] = name
    tool_call_id = getattr(message, "tool_call_id", None)
    if tool_call_id:
        result["tool_call_id"] = tool_call_id
    return result


def _serialise_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return _short_text(value) if isinstance(value, str) else value
    if isinstance(value, list):
        if value and hasattr(value[0], "content"):
            return [_serialise_message(item) for item in value]
        return [_serialise_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _serialise_value(item) for key, item in value.items()}
    if hasattr(value, "content"):
        return _serialise_message(value)
    return _short_text(value)


def _serialise_state(state: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _serialise_value(value) for key, value in state.items()}
