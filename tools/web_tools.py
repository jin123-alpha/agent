import json
import re
import urllib.parse
import urllib.request
from html.parser import HTMLParser

# ── 搜索源配置（按优先级排列，依次尝试）──────────────────────────────
# Bing 国内外均可访问；DuckDuckGo 作为国际备用
SEARCH_URLS = [
    {
        "name": "bing",
        "url": "https://www.bing.com/search",
        "method": "GET",
    },
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

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)
MAX_FETCH_RESULTS = 5
MAX_PAGE_BYTES = 1_000_000


# ═══════════════════════════════════════════════════════════════════════
# Bing HTML 解析器
# ═══════════════════════════════════════════════════════════════════════
class BingHTMLParser(HTMLParser):
    """解析 Bing 搜索结果页（www.bing.com / cn.bing.com 通用）"""

    def __init__(self):
        super().__init__()
        self.results = []
        self.current_result = None
        self._in_link = False
        self._in_snippet = False
        self._link_href = ""
        self._title_parts = []
        self._snippet_parts = []
        self._capture_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)

        if tag in {"script", "style", "noscript", "svg"}:
            self._capture_depth += 1
            return

        # Bing 搜索结果标题链接: <a> 在 <h2> 内, 或带特定 class
        if tag == "a" and self._capture_depth == 0:
            href = attrs_dict.get("href", "")
            if href and not href.startswith("javascript:"):
                # 可能是结果链接
                self._in_link = True
                self._link_href = href
                self._title_parts = []

        # Bing 摘要通常在 <p> 或 class 含 snippet 的元素中
        if tag == "p" and self._capture_depth == 0:
            cls = attrs_dict.get("class", "")
            if any(k in cls.lower() for k in ("snippet", "b_lineclamp", "b_caption")):
                self._in_snippet = True
                self._snippet_parts = []
        elif tag == "span" and self._capture_depth == 0:
            cls = attrs_dict.get("class", "")
            if "b_caption" in cls.lower():
                self._in_snippet = True
                self._snippet_parts = []

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"}:
            if self._capture_depth > 0:
                self._capture_depth -= 1
            return

        if tag == "a" and self._in_link:
            self._in_link = False
            title = " ".join(self._title_parts).strip()
            url = self._clean_bing_url(self._link_href)
            if title and url and self._looks_like_result(url):
                self.current_result = {
                    "title": title,
                    "url": url,
                    "snippet": "",
                }
            self._title_parts = []
            self._link_href = ""

        if (tag == "p" or tag == "span") and self._in_snippet:
            self._in_snippet = False
            snippet = " ".join(self._snippet_parts).strip()
            if self.current_result and snippet:
                self.current_result["snippet"] = snippet
                self.results.append(self.current_result)
                self.current_result = None
            self._snippet_parts = []

    def handle_data(self, data):
        if self._capture_depth > 0:
            return

        text = data.strip()
        if not text:
            return

        if self._in_link:
            self._title_parts.append(text)
        if self._in_snippet:
            self._snippet_parts.append(text)

    @staticmethod
    def _clean_bing_url(url: str) -> str:
        """去掉 Bing 跳转包装，还原真实 URL"""
        if not url:
            return ""
        if url.startswith("//"):
            url = f"https:{url}"
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)
        # Bing 有时把真实 URL 放在 query 参数里
        for key in ("u", "url", "q"):
            if key in query:
                return query[key][0]
        return url

    @staticmethod
    def _looks_like_result(url: str) -> bool:
        """排除导航链接、Bing 自身页面等"""
        skip_domains = {"bing.com", "microsoft.com/bing", "go.microsoft.com"}
        parsed = urllib.parse.urlparse(url)
        netloc = parsed.netloc.lower()
        for skip in skip_domains:
            if skip in netloc:
                return False
        return True


# ═══════════════════════════════════════════════════════════════════════
# DuckDuckGo HTML 解析器（保留）
# ═══════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════════════

def _get_parser(source_name: str):
    """根据源名称返回对应的解析器"""
    if source_name == "bing":
        return BingHTMLParser()
    return DuckDuckGoHTMLParser()


def build_search_request(source: dict, query: str):
    encoded_query = urllib.parse.urlencode({"q": query})
    method = source["method"]
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }

    if method == "POST":
        return urllib.request.Request(
            source["url"],
            data=encoded_query.encode("utf-8"),
            headers={
                **headers,
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )

    return urllib.request.Request(
        f"{source['url']}?{encoded_query}",
        headers=headers,
    )


def search_with_source(source: dict, query: str) -> list[dict]:
    request = build_search_request(source, query)

    with urllib.request.urlopen(request, timeout=15) as response:
        html = response.read().decode("utf-8", errors="replace")

    parser = _get_parser(source["name"])
    parser.feed(html)

    # DuckDuckGo 解析器需要手动触发最后一条
    if hasattr(parser, "append_current_result"):
        parser.append_current_result()

    return [
        result
        for result in parser.results
        if result["title"] and result["url"]
    ]


# ═══════════════════════════════════════════════════════════════════════
# 页面抓取
# ═══════════════════════════════════════════════════════════════════════

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
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
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


# ═══════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════

def web_search(
    query: str,
    max_results: int = 5,
    fetch_content: bool = True,
    content_chars: int = 1200,
) -> str:
    """
    web_search(query, max_results=5, fetch_content=True, content_chars=1200)

    联网搜索网页内容，返回标题、链接、搜索摘要，并默认抓取每个结果网页的正文片段。

    搜索源按优先级依次尝试: Bing → DuckDuckGo Lite → DuckDuckGo HTML
    Bing 在国内外均可访问，是国内网络环境下的首选。
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
