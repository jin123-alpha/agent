import base64
import json
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

def fetch_readme(repo_full_name: str, max_chars: int = 15000) -> str:
    """
    fetch_readme(repo_full_name: str, max_chars: int = 15000) -> str
    获取指定 GitHub 仓库的 README 内容。
    为了防止上下文爆炸，超过 max_chars 的内容会被截断，并附上截断提示。
    """
    token = os.environ.get("GITHUB_API_KEY", "")
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubSearchAgent/1.0",
    }
    if token:
        headers["Authorization"] = f"token {token}"
        
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                content = data.get("content", "")
                if content:
                    text = base64.b64decode(content).decode("utf-8", errors="replace")
                    if len(text) > max_chars:
                        return text[:max_chars] + "\n\n... (README 已截断)"
                    return text
    except Exception as e:
        logger.debug(f"fetch_readme {repo_full_name} failed: {e}")
        return f"无法获取 {repo_full_name} 的 README，错误：{e}"
    
    return ""
