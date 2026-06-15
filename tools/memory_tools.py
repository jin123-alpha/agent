from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MEMORY_PATH = PROJECT_ROOT / "data" / "memory.json"
MAX_MEMORY_ITEMS = 200
MAX_CONTENT_CHARS = 1000


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_text(value: str | None, default: str = "") -> str:
    if value is None:
        return default

    return str(value).strip()


def load_memory_items() -> list[dict]:
    if not MEMORY_PATH.exists():
        return []

    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    items = []
    for item in data:
        if not isinstance(item, dict):
            continue

        content = normalize_text(item.get("content"))
        if not content:
            continue

        memory_id = normalize_text(item.get("id")) or uuid.uuid4().hex[:12]
        category = normalize_text(item.get("category"), "general") or "general"
        created_at = normalize_text(item.get("created_at")) or utc_now()
        updated_at = normalize_text(item.get("updated_at")) or created_at

        items.append({
            "id": memory_id,
            "category": category,
            "content": content[:MAX_CONTENT_CHARS],
            "created_at": created_at,
            "updated_at": updated_at,
        })

    return items[-MAX_MEMORY_ITEMS:]


def save_memory_items(items: list[dict]) -> None:
    MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    MEMORY_PATH.write_text(
        json.dumps(items[-MAX_MEMORY_ITEMS:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def item_matches(item: dict, query: str, category: str) -> bool:
    if category and item["category"].lower() != category.lower():
        return False

    if not query:
        return True

    haystack = f"{item['category']} {item['content']}".lower()
    return query.lower() in haystack


def remember(content: str, category: str = "general") -> str:
    """
    remember(content, category="general")：保存一条长期记忆，适合记录用户明确要求记住的偏好、事实、项目约定或长期指令；不要保存密码、API key、令牌等敏感秘密。
    """
    content = normalize_text(content)
    category = normalize_text(category, "general") or "general"

    if not content:
        return "记忆内容不能为空。"

    if len(content) > MAX_CONTENT_CHARS:
        content = content[:MAX_CONTENT_CHARS]

    items = load_memory_items()
    now = utc_now()

    for item in items:
        if item["category"].lower() == category.lower() and item["content"] == content:
            item["updated_at"] = now
            save_memory_items(items)
            return json.dumps(
                {
                    "ok": True,
                    "action": "updated_existing",
                    "memory": item,
                },
                ensure_ascii=False,
                indent=2,
            )

    item = {
        "id": uuid.uuid4().hex[:12],
        "category": category,
        "content": content,
        "created_at": now,
        "updated_at": now,
    }
    items.append(item)
    save_memory_items(items)

    return json.dumps(
        {
            "ok": True,
            "action": "created",
            "memory": item,
        },
        ensure_ascii=False,
        indent=2,
    )


def recall_memory(query: str = "", category: str = "", max_results: int = 10) -> str:
    """
    recall_memory(query="", category="", max_results=10)：查询长期记忆；query 为空时返回最近记忆，可按 category 精确过滤。
    """
    query = normalize_text(query)
    category = normalize_text(category)

    if max_results < 1:
        max_results = 1
    elif max_results > 50:
        max_results = 50

    items = load_memory_items()
    matches = [
        item
        for item in reversed(items)
        if item_matches(item, query, category)
    ][:max_results]

    return json.dumps(
        {
            "ok": True,
            "count": len(matches),
            "memories": matches,
        },
        ensure_ascii=False,
        indent=2,
    )


def forget_memory(memory_id: str) -> str:
    """
    forget_memory(memory_id)：根据记忆 id 删除一条长期记忆；删除前可先调用 recall_memory 查找 id。
    """
    memory_id = normalize_text(memory_id)
    if not memory_id:
        return "memory_id 不能为空。"

    items = load_memory_items()
    kept_items = [item for item in items if item["id"] != memory_id]

    if len(kept_items) == len(items):
        return f"没有找到记忆：{memory_id}"

    save_memory_items(kept_items)
    return json.dumps(
        {
            "ok": True,
            "deleted_id": memory_id,
            "remaining_count": len(kept_items),
        },
        ensure_ascii=False,
        indent=2,
    )


def format_memory_for_prompt(max_items: int = 20) -> str:
    items = load_memory_items()
    if not items:
        return "暂无长期记忆。"

    selected_items = list(reversed(items))[:max_items]
    lines = []
    for item in selected_items:
        lines.append(f"- [{item['id']}] {item['category']}：{item['content']}")

    return "\n".join(lines)
