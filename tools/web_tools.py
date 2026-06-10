import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser


SEARCH_URLS = [
    {
        "name": "duckduckgo_lite",
        "url": "https://lite.duckduckgo.com/lite/",
        "method": "GET",
    },
    {
        "name": "duckduckgo_html",
        "url": "https://html.duckduckgo.com/html/",
        "method": "POST",
    },
]
USER_AGENT = "Mozilla/5.0 (compatible; SimpleAgent/1.0)"
MAX_FETCH_RESULTS = 5
MAX_PAGE_BYTES = 1_000_000


class DuckDuckGoHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self.current_field = None

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        class_name = attrs.get("class", "")

        if tag == "a" and ("result-link" in class_name or "result__a" in class_name):
            self.append_current_result()
            self.current_result = {
                "title": "",
                "url": self.clean_duckduckgo_url(attrs.get("href", "")),
                "snippet": "",
            }
            self.current_field = "title"
        elif self.current_result and (
            "result-snippet" in class_name
            or "result__snippet" in class_name
        ):
            self.current_field = "snippet"

    def handle_data(self, data):
        if not self.current_result or not self.current_field:
            return

        text = data.strip()
        if not text:
            return

        previous = self.current_result[self.current_field]
        separator = " " if previous else ""
        self.current_result[self.current_field] = f"{previous}{separator}{text}"

    def handle_endtag(self, tag):
        if tag == "a" and self.current_field == "title":
            self.current_field = None
        elif self.current_field == "snippet":
            self.current_field = None

    def append_current_result(self):
        if self.current_result and self.current_result["title"]:
            self.results.append(self.current_result)
            self.current_result = None
            self.current_field = None

    def clean_duckduckgo_url(self, url: str) -> str:
        if not url:
            return ""

        if url.startswith("//"):
            url = f"https:{url}"

        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)

        if "uddg" in query:
            return query["uddg"][0]

        return url


def build_search_request(source: dict, query: str):
    encoded_query = urllib.parse.urlencode({"q": query})
    method = source["method"]

    if method == "POST":
        return urllib.request.Request(
            source["url"],
            data=encoded_query.encode("utf-8"),
            headers={
                "User-Agent": USER_AGENT,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

    return urllib.request.Request(
        f"{source['url']}?{encoded_query}",
        headers={
            "User-Agent": USER_AGENT,
        },
    )


def search_with_source(source: dict, query: str) -> list[dict]:
    request = build_search_request(source, query)

    with urllib.request.urlopen(request, timeout=10) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = DuckDuckGoHTMLParser()
    parser.feed(html)
    parser.append_current_result()

    return [
        result
        for result in parser.results
        if result["title"] and result["url"]
    ]


class PageTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if self.skip_depth:
            return

        text = data.strip()
        if text:
            self.parts.append(text)

    def get_text(self) -> str:
        text = " ".join(self.parts)
        return re.sub(r"\s+", " ", text).strip()


def fetch_page_content(url: str, max_chars: int = 1200) -> str:
    if max_chars < 200:
        max_chars = 200
    elif max_chars > 5000:
        max_chars = 5000

    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type and "text/plain" not in content_type:
                return f"跳过非文本内容：{content_type or '未知类型'}"

            raw = response.read(MAX_PAGE_BYTES)
    except Exception as exc:
        return f"页面读取失败：{exc}"

    html = raw.decode("utf-8", errors="replace")
    parser = PageTextParser()
    parser.feed(html)
    content = parser.get_text()

    if not content:
        return "未提取到页面正文。"

    if len(content) > max_chars:
        return f"{content[:max_chars]}..."

    return content


def web_search(query: str, max_results: int = 5, fetch_content: bool = True, content_chars: int = 1200) -> str:
    """
    web_search(query, max_results=5, fetch_content=True, content_chars=1200)：联网搜索网页内容，返回标题、链接、搜索摘要，并默认抓取每个结果网页的正文片段。
    """
    if not query.strip():
        return "搜索关键词不能为空。"

    if max_results < 1:
        max_results = 1
    elif max_results > 10:
        max_results = 10

    attempted_sources = []
    last_error = None
    results = []
    source_name = None

    for source in SEARCH_URLS:
        source_name = source["name"]
        attempted_sources.append(source_name)

        try:
            results = search_with_source(source, query)
        except Exception as exc:
            last_error = f"{source_name}: {exc}"
            continue

        if results:
            break

        last_error = f"{source_name}: 没有找到搜索结果"

    if not results:
        return json.dumps(
            {
                "query": query,
                "ok": False,
                "attempted_sources": attempted_sources,
                "error": last_error or "没有找到搜索结果。",
            },
            ensure_ascii=False,
            indent=2,
        )

    results = results[:max_results]

    if fetch_content:
        for result in results[:MAX_FETCH_RESULTS]:
            result["content"] = fetch_page_content(result["url"], content_chars)

    return json.dumps(
        {
            "query": query,
            "source": source_name,
            "attempted_sources": attempted_sources,
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
