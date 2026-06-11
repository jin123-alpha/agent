def content_to_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)

        return "".join(parts)

    return ""


def message_content_text(message) -> str:
    return content_to_text(getattr(message, "content", ""))


def message_id(message):
    return getattr(message, "id", None)


def final_message_content(messages) -> str:
    for message in reversed(messages):
        content = message_content_text(message)
        if content.strip():
            return content

    return ""
