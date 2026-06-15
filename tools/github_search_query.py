from __future__ import annotations

"""convert_query 节点：将 RequirementAgent 输出的结构化 JSON 转为 GitHub Search API 的 q 参数。"""

import json
import logging
import re

logger = logging.getLogger(__name__)


def _extract_language(data: dict) -> str:
    """从需求数据中提取语言过滤字段。"""
    language = data.get("language", "")
    if isinstance(language, str) and language.strip() and language.strip().lower() != "不限":
        return language.strip().lower()
    return ""


def _build_searchable_query(input_json: str) -> dict:
    """
    从 RequirementAgent 的 JSON 输出构建 GitHub Search API 的 q 参数。

    输入示例:
    {
        "keywords": ["RAG", "PDF", "Ollama", "Web UI"],
        "language": "Python",
        "constraints": [...],
        "preferences": [...]
    }

    输出: 'RAG PDF Ollama "Web UI" language:python'
    """
    try:
        data = json.loads(input_json) if isinstance(input_json, str) else input_json
    except (json.JSONDecodeError, TypeError):
        logger.warning("convert_query: 输入不是有效 JSON，直接作为关键词使用")
        return input_json.strip()

    if not isinstance(data, dict):
        return str(data).strip()

    # 关键词：含空格或特殊字符的用引号包裹
    keywords = data.get("keywords", [])
    if isinstance(keywords, str):
        keywords = [keywords]

    q_parts: list[str] = []

    # project_type 作为第一个搜索词（描述用户意图的核心信号）
    project_type = data.get("project_type", "")
    if isinstance(project_type, str) and project_type.strip():
        q_parts.append(project_type.strip())

    for kw in keywords:
        kw = str(kw).strip()
        if not kw:
            continue
        q_parts.append(kw)

    return {
        "searchable_query": ":".join(q_parts),
        "target_language": _extract_language(data),
    }


def convert_query(state: dict, config: dict | None = None) -> dict:
    """
    LangGraph 节点：从 messages 中提取输入 JSON，转为 searchable_query。

    期望 state 包含:
    - messages: [HumanMessage(content=<JSON>), ...]
    - searchable_query: str (由本节点写入)
    - user_query: str (原始结构化 JSON，供后续节点使用)
    """
    # 从 messages 提取输入
    messages = state.get("messages", [])
    if not messages:
        logger.warning("convert_query: messages 为空")
        return {"searchable_query": "", "user_query": ""}

    input_text = messages[-1].content if hasattr(messages[-1], "content") else str(messages[-1])

    # 解析 prompt 包装（_inject_context 宏可能包裹了 JSON）
    # 格式1: 直接 JSON（test.py mock input）
    # 格式2: "## 前序 Agent 传递的上下文数据\n```json\n{AgentName: json_string}\n```\n..."
    if "## 前序 Agent 传递的上下文数据" in input_text:
        match = re.search(r"```json\s*(.*?)\s*```", input_text, re.DOTALL)
        if match:
            input_text = match.group(1).strip()

    # 如果解析出来的是 {AgentName: "escaped JSON string"} 格式，取第一个值
    try:
        data = json.loads(input_text) if isinstance(input_text, str) else input_text
        if isinstance(data, dict):
            # 检查是否是 _inject_context 的嵌套格式
            values = list(data.values())
            if values and all(isinstance(v, str) for v in values):
                # 取第一个 Agent 的输出字符串，可能是 JSON
                first_val = values[0]
                try:
                    nested = json.loads(first_val)
                    if isinstance(nested, dict):
                        input_text = first_val
                except (json.JSONDecodeError, TypeError):
                    pass
    except (json.JSONDecodeError, TypeError):
        pass

    result = _build_searchable_query(input_text)
    searchable_query = result["searchable_query"]
    target_language = result["target_language"]
    logger.info(f"convert_query: searchable_query = {searchable_query}, target_language = {target_language}")

    return {
        "searchable_query": searchable_query,
        "target_language": target_language,
        "user_query": input_text,
    }
