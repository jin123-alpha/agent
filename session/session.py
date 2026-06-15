from __future__ import annotations

from dataclasses import dataclass, field

from tools.memory_tools import recall_memory, remember


@dataclass
class Session:
    """
    Owns conversational/session memory integration and caching.
    """

    id: str = "default"
    metadata: dict = field(default_factory=dict)
    
    # 状态缓存，避免重复请求和分析
    _repo_analysis_cache: dict[str, dict] = field(default_factory=dict)
    _readme_cache: dict[str, str] = field(default_factory=dict)

    def remember(self, content: str, category: str = "general") -> str:
        return remember(content, category)

    def recall(self, query: str = "", category: str = "", max_results: int = 10) -> str:
        return recall_memory(query, category, max_results)

    def get_cached_readme(self, repo_full_name: str) -> str | None:
        """获取缓存的 README 内容"""
        return self._readme_cache.get(repo_full_name)

    def set_cached_readme(self, repo_full_name: str, content: str):
        """缓存 README 内容"""
        self._readme_cache[repo_full_name] = content

    def get_repo_analysis(self, repo_full_name: str) -> dict | None:
        """获取已缓存的仓库结构/特性分析"""
        return self._repo_analysis_cache.get(repo_full_name)

    def set_repo_analysis(self, repo_full_name: str, analysis: dict):
        """缓存仓库特性分析结果"""
        self._repo_analysis_cache[repo_full_name] = analysis
