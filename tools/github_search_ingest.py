"""ingest_github_repos 节点：调用 GitHub Search API 搜索仓库，并发抓取 README 和文档。"""

import asyncio
import base64
import logging
import os
from pathlib import Path
from typing import Callable

import httpx

logger = logging.getLogger(__name__)


def _noop(_message: str) -> None:
    """默认空进度回调。"""
    return None

# 抑制 httpx 的 HTTP 请求日志
logging.getLogger("httpx").setLevel(logging.WARNING)

FILE_CONTENT_CACHE: dict[str, str] = {}


def _get_github_token(config: dict | None = None) -> str:
    """从 config.json 读取 github_api_key，fallback 到环境变量 GITHUB_API_KEY。"""
    cfg = config or {}
    token = cfg.get("github_api_key", "")
    if token:
        return token
    return os.environ.get("GITHUB_API_KEY", "")


def _build_headers(token: str) -> dict:
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "GitHubSearchAgent/1.0",
    }


async def _fetch_readme_content(
    repo_full_name: str, headers: dict, client: httpx.AsyncClient
) -> str:
    """获取仓库 README 内容（base64 解码）。"""
    url = f"https://api.github.com/repos/{repo_full_name}/readme"
    try:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            readme_data = response.json()
            content = readme_data.get("content", "")
            if content:
                return base64.b64decode(content).decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug(f"fetch_readme {repo_full_name}: {exc}")
    return ""


async def _fetch_file_content(
    download_url: str, client: httpx.AsyncClient
) -> str:
    """获取单个文件内容（带缓存）。"""
    if download_url in FILE_CONTENT_CACHE:
        return FILE_CONTENT_CACHE[download_url]
    try:
        response = await client.get(download_url)
        if response.status_code == 200:
            text = response.text
            FILE_CONTENT_CACHE[download_url] = text
            return text
    except Exception as exc:
        logger.debug(f"fetch_file {download_url}: {exc}")
    return ""


async def _fetch_directory_markdown(
    repo_full_name: str, path: str, headers: dict, client: httpx.AsyncClient
) -> str:
    """获取目录下所有 .md 文件内容。"""
    md_parts: list[str] = []
    url = f"https://api.github.com/repos/{repo_full_name}/contents/{path}"
    try:
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            items = response.json()
            tasks = []
            for item in items:
                if item["type"] == "file" and item["name"].lower().endswith(".md"):
                    tasks.append(
                        _fetch_file_content(item["download_url"], client)
                    )
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for item, content in zip(items, results):
                    if (
                        item["type"] == "file"
                        and item["name"].lower().endswith(".md")
                        and not isinstance(content, Exception)
                    ):
                        md_parts.append(f"\n\n# {item['name']}\n{content}")
    except Exception as exc:
        logger.debug(f"fetch_directory_markdown {repo_full_name}/{path}: {exc}")
    return "".join(md_parts)


async def _fetch_repo_documentation(
    repo_full_name: str, headers: dict, client: httpx.AsyncClient
) -> str:
    """并发抓取 README + 根目录 .md 文件 + docs/ 目录。"""
    readme_task = asyncio.create_task(
        _fetch_readme_content(repo_full_name, headers, client)
    )

    doc_parts: list[str] = []
    root_url = f"https://api.github.com/repos/{repo_full_name}/contents"
    try:
        response = await client.get(root_url, headers=headers)
        if response.status_code == 200:
            items = response.json()
            tasks = []
            for item in items:
                if item["type"] == "file" and item["name"].lower().endswith(".md"):
                    if item["name"].lower() != "readme.md":
                        tasks.append(
                            asyncio.create_task(
                                _fetch_file_content(item["download_url"], client)
                            )
                        )
                elif item["type"] == "dir" and item["name"].lower() in (
                    "docs",
                    "documentation",
                ):
                    tasks.append(
                        asyncio.create_task(
                            _fetch_directory_markdown(
                                repo_full_name, item["name"], headers, client
                            )
                        )
                    )
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if not isinstance(res, Exception) and res:
                    doc_parts.append(res)
    except Exception as exc:
        logger.debug(f"fetch_repo_documentation {repo_full_name}: {exc}")

    readme = await readme_task
    if readme:
        doc_parts.insert(0, f"# README\n{readme}")

    return "".join(doc_parts) if doc_parts else ""


async def _fetch_github_repositories(
    query: str,
    max_results: int,
    per_page: int,
    headers: dict,
    progress: dict | None = None,
    report: Callable[[str], None] = _noop,
) -> list[dict]:
    """调用 GitHub Search API 搜索仓库列表。"""
    url = "https://api.github.com/search/repositories"
    repos: list[dict] = []
    num_pages = max(1, max_results // per_page)

    async with httpx.AsyncClient(timeout=30.0) as client:
        for page in range(1, num_pages + 1):
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page,
            }
            try:
                response = await client.get(url, headers=headers, params=params)

                if response.status_code in (403, 429):
                    logger.warning(
                        f"GitHub API rate limited (page {page}, status {response.status_code})"
                    )
                    retry_after = response.headers.get("Retry-After", "30")
                    try:
                        wait = min(int(retry_after), 60)
                    except ValueError:
                        wait = 30
                    logger.info(f"Waiting {wait}s for rate limit reset...")
                    await asyncio.sleep(wait)
                    response = await client.get(
                        url, headers=headers, params=params
                    )
                    if response.status_code != 200:
                        logger.error(
                            f"GitHub API error after retry: {response.status_code}"
                        )
                        break

                if response.status_code != 200:
                    try:
                        msg = response.json().get("message", response.text[:200])
                    except Exception:
                        msg = response.text[:200]
                    logger.error(
                        f"GitHub API error {response.status_code}: {msg}"
                    )
                    break

                data = response.json()
                items = data.get("items", [])

                if not items:
                    break

                # 并发抓取每个仓库的文档
                async def _fetch_with_progress(full_name: str) -> str:
                    doc = await _fetch_repo_documentation(full_name, headers, client)
                    if progress is not None:
                        progress["done"] += 1
                        report(
                            f"正在抓取仓库文档 {progress['done']}/{progress['total']}"
                            f" · {full_name}"
                        )
                    return doc

                tasks = []
                for repo in items:
                    full_name = repo.get("full_name", "")
                    tasks.append(
                        asyncio.create_task(_fetch_with_progress(full_name))
                    )
                docs = await asyncio.gather(*tasks, return_exceptions=True)

                for repo, doc in zip(items, docs):
                    repo_link = repo["html_url"]
                    full_name = repo.get("full_name", "")
                    star_count = repo.get("stargazers_count", 0)
                    license_info = repo.get("license")
                    license_spdx = license_info.get("spdx_id", "") if license_info else ""
                    repos.append(
                        {
                            "title": repo.get("name", ""),
                            "full_name": full_name,
                            "html_url": repo_link,
                            "description": repo.get("description", ""),
                            "stars": star_count,
                            "license": license_spdx,
                            "language": repo.get("language", ""),
                            "updated_at": repo.get("updated_at", ""),
                            "open_issues_count": repo.get("open_issues_count", 0),
                            "combined_doc": (
                                doc if not isinstance(doc, Exception) else ""
                            ),
                        }
                    )
            except Exception as exc:
                logger.debug(
                    f"fetch_github_repositories page {page}: {type(exc).__name__}: {exc}"
                )
                break

    logger.info(f"Fetched {len(repos)} repositories for query '{query}'.")
    return repos


async def _ingest_github_repos_async(
    state: dict,
    config: dict | None = None,
    report: Callable[[str], None] = _noop,
) -> dict:
    """异步核心：按冒号分隔关键词，每个词单独搜索后去重合并。"""
    cfg = config or {}
    searchable_query = state.get("searchable_query", "")
    user_query = state.get("user_query", "")

    if not searchable_query:
        logger.warning("ingest_github_repos: searchable_query 为空")
        return {
            "repositories": [],
            "user_query": user_query or searchable_query,
        }

    token = _get_github_token(cfg)
    headers = _build_headers(token)

    max_results = cfg.get("github_max_results", 100)
    per_page = cfg.get("github_per_page", 25)

    # 拆分关键词（language 由 convert_query 独立提供，不在此处）
    keywords = [p.strip() for p in searchable_query.split(":") if p.strip()]
    target_language = state.get("target_language", "")

    if not keywords:
        keywords = searchable_query.split(":")
        keywords = [p.strip() for p in keywords if p.strip()]

    report(f"正在搜索 {len(keywords)} 个关键词：{('、'.join(keywords))[:60]}")

    # 共享进度计数器（跨并发的关键词搜索累计已抓取文档数）
    progress = {"done": 0, "total": 0}

    # 并发搜索每个关键词
    tasks = []
    for kw in keywords:
        query = f"{kw} {target_language}".strip() if target_language else kw
        tasks.append(
            asyncio.create_task(
                _fetch_github_repositories(
                    query, max_results, per_page, headers, progress, report
                )
            )
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 去重合并
    seen: set[str] = set()
    all_repos: list[dict] = []
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"ingest_github_repos keyword error: {result}")
            continue
        for repo in result:
            fn = repo.get("full_name", "")
            if fn and fn not in seen:
                seen.add(fn)
                all_repos.append(repo)

    # 按 stars 降序排列，预截断（减少后续语义检索负担）
    all_repos.sort(key=lambda x: x.get("stars", 0), reverse=True)
    max_repos_before_retrieval = cfg.get("github_max_results", 100) * 2  # 默认 200
    if len(all_repos) > max_repos_before_retrieval:
        logger.info(f"ingest: pre-truncating {len(all_repos)} → {max_repos_before_retrieval}")
        all_repos = all_repos[:max_repos_before_retrieval]

    logger.info(f"Ingested {len(all_repos)} unique repositories from {len(keywords)} keywords.")

    return {
        "repositories": all_repos,
        "user_query": user_query or searchable_query,
    }


def ingest_github_repos(
    state: dict,
    config: dict | None = None,
    report: Callable[[str], None] = _noop,
) -> dict:
    """LangGraph 节点入口（同步包装）。"""
    try:
        return asyncio.run(_ingest_github_repos_async(state, config, report))
    except RuntimeError:
        # 已有 event loop 的情况（LangGraph 可能已有）
        import nest_asyncio

        nest_asyncio.apply()
        return asyncio.run(_ingest_github_repos_async(state, config, report))
