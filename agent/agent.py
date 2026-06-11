from dataclasses import dataclass, field
from typing import Any

from guardrail import Guardrail
from handoff import Handoff
from session import Session
from tools import ToolRegistry, create_default_tool_registry
from tracing import Tracer


DEFAULT_AGENT_INSTRUCTIONS = """
你是一个简单 Agent。

记忆使用规则：
- 当用户明确要求你记住某个偏好、事实、项目约定或长期指令时，调用 remember。
- 当用户询问以前记住了什么，或当前问题可能依赖长期记忆时，调用 recall_memory。
- 当用户要求忘记某条记忆时，先查找对应记忆，再调用 forget_memory。
- 不要把密码、API key、访问令牌等敏感秘密写入长期记忆。
- 不要默认把长期记忆放入上下文；只有需要时才调用 recall_memory。

需要外部信息、读写文件、查询记忆或保存记忆时，选择最合适的工具。
不要在最终回答中直接输出大段代码；需要创建或修改文件时使用文件工具。

如果不需要工具，直接回答。
"""


@dataclass
class Agent:
    """
    Describes an agent's capabilities, instructions, tools, and collaboration hooks.
    """

    name: str
    instructions: str
    tools: ToolRegistry = field(default_factory=create_default_tool_registry)
    handoffs: list[Handoff] = field(default_factory=list)
    guardrails: list[Guardrail] = field(default_factory=list)
    session: Session | None = None
    tracer: Tracer | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def system_prompt(self) -> str:
        tool_names = ", ".join(self.tools.names())
        return f"{self.instructions.strip()}\n\n当前可用工具名：{tool_names}"


def create_default_agent() -> Agent:
    return Agent(
        name="simple-tool-agent",
        instructions=DEFAULT_AGENT_INSTRUCTIONS,
        session=Session(),
    )
