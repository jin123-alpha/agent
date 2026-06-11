from dataclasses import dataclass
from typing import Callable


@dataclass
class ToolRegistry:
    """
    Holds executable tool functions and adapts them for LangChain.
    """

    tools: list[Callable]

    def names(self) -> list[str]:
        return [tool.__name__ for tool in self.tools]

    def as_langchain_tools(self, StructuredTool):
        return [
            StructuredTool.from_function(
                func=tool,
                name=tool.__name__,
                description=(tool.__doc__ or "").strip() or None,
            )
            for tool in self.tools
        ]


def create_default_tool_registry() -> ToolRegistry:
    from . import TOOLS

    return ToolRegistry(tools=list(TOOLS))
