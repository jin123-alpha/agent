from dataclasses import dataclass, field

from tools.memory_tools import recall_memory, remember


@dataclass
class Session:
    """
    Owns conversational/session memory integration.
    """

    id: str = "default"
    metadata: dict = field(default_factory=dict)

    def remember(self, content: str, category: str = "general") -> str:
        return remember(content, category)

    def recall(self, query: str = "", category: str = "", max_results: int = 10) -> str:
        return recall_memory(query, category, max_results)
