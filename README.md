# Simple Tool Agent

这是一个基于 LangGraph + LangChain 的最小可扩展 Agent 项目。`run.py` 负责构建 `StateGraph`，用模型节点和工具节点组成循环；`tools/` 目录负责放置工具函数。新增工具时，只需要定义函数、写好 docstring，然后加入 `tools/__init__.py` 的 `TOOLS` 数组。

## 文件结构

```text
agent/
|-- config.json             # 模型配置：base_url、api_key、model、stream
|-- data/
|   `-- memory.json         # 本地长期记忆文件，默认被 .gitignore 忽略
|-- requirements.txt        # LangGraph / LangChain 依赖
|-- run.py                  # 项目入口：构建 LangGraph、执行模型节点和工具节点
|-- tools/
|   |-- __init__.py         # 工具注册表：TOOLS 和 TOOL_MAP
|   |-- code_tools.py       # 文件结构、读文件、创建文件、编辑文件、编译检查
|   |-- math_tools.py       # 数学工具
|   |-- memory_tools.py     # 长期记忆工具
|   |-- security_tools.py   # 随机密钥工具
|   `-- web_tools.py        # 网络搜索和网页正文抓取工具
`-- README.md
```

## 运行

先安装依赖：

```bash
pip install -r requirements.txt
```

然后运行：

```bash
python run.py
```

`run_agent(user_input, max_tool_calls=10)` 会把用户输入放入 LangGraph 状态，然后在 `agent -> tools -> agent` 图中循环。模型通过 LangChain 的 `bind_tools()` 选择工具；工具执行结果会以 `ToolMessage` 返回给图，直到模型给出最终回答。

当 `stream=true` 时，`run_agent()` 使用 LangGraph 的 `messages` stream mode 流式输出模型可见文本。

## 当前工具

### `add`

计算两个整数之和。

```json
{"tool": "add", "arguments": {"a": 3, "b": 5}}
```

### `generate_runtime_secret`

生成运行时随机密钥。

```json
{"tool": "generate_runtime_secret", "arguments": {"length": 32}}
```

### `remember`

保存一条长期记忆，适合记录用户明确要求记住的偏好、事实、项目约定或长期指令。记忆存储在 `data/memory.json`，默认不会提交到 Git。

```json
{"tool": "remember", "arguments": {"content": "用户喜欢简洁的中文回答", "category": "preference"}}
```

### `recall_memory`

查询长期记忆。`query` 为空时返回最近记忆，可按 `category` 精确过滤。

```json
{"tool": "recall_memory", "arguments": {"query": "中文回答", "category": "preference", "max_results": 5}}
```

### `forget_memory`

根据记忆 id 删除一条长期记忆。删除前可先调用 `recall_memory` 查找 id。

```json
{"tool": "forget_memory", "arguments": {"memory_id": "abc123def456"}}
```

### `list_workspace_files`

获取当前工作区文件结构树，默认忽略 `.git`、`.agents`、`.codex`、`__pycache__`。

```json
{"tool": "list_workspace_files", "arguments": {"max_depth": 5}}
```

### `read_code_file`

读取项目内文件的指定行范围，返回带行号的代码。

```json
{"tool": "read_code_file", "arguments": {"path": "run.py", "start_line": 1, "end_line": 80}}
```

### `create_file`

在项目目录内创建文本文件。默认不覆盖已存在文件，可自动创建父目录。

```json
{"tool": "create_file", "arguments": {"path": "notes/todo.md", "content": "# TODO\n", "overwrite": false}}
```

### `edit_code_file`

精确替换项目内文本文件中的内容。默认要求只替换 1 处，避免误改多个位置。

```json
{"tool": "edit_code_file", "arguments": {"path": "run.py", "old_text": "旧内容", "new_text": "新内容", "expected_replacements": 1}}
```

### `compile_python_files`

编译检查 Python 文件并返回错误信息。`paths` 为空时检查整个工作区所有 `.py` 文件。

```json
{"tool": "compile_python_files", "arguments": {"paths": ["run.py", "tools/code_tools.py"]}}
```

### `web_search`

联网搜索网页内容，返回标题、链接、搜索摘要，并默认抓取搜索结果网页的正文片段。搜索源不可用时会自动尝试下一个源。

```json
{"tool": "web_search", "arguments": {"query": "OpenAI API", "max_results": 3, "fetch_content": true, "content_chars": 1200}}
```

## 添加新工具

1. 在 `tools/` 下创建或选择一个工具文件，例如 `tools/date_tools.py`。
2. 定义普通 Python 函数。
3. 在函数 docstring 中（可以模仿已有函数的注释）写上给 system prompt 的工具说明。
4. 在 `tools/__init__.py` 中 import 函数，并加入 `TOOLS` 数组。

示例：

```python
# tools/date_tools.py
from datetime import datetime


def current_time() -> str:
    """
    current_time()：获取当前系统时间。
    """
    return datetime.now().isoformat(timespec="seconds")
```

注册：

```python
from .date_tools import current_time

TOOLS = [
    current_time,
]
```

`run.py` 会把 `TOOLS` 中的普通 Python 函数包装成 LangChain `StructuredTool`，并通过 `ChatOpenAI(...).bind_tools(...)` 交给模型。`tools/__init__.py` 只负责维护工具注册表和分发映射。

## 长期记忆

长期记忆保存在 `data/memory.json`。为了避免 system prompt 过长，程序不会默认把长期记忆全文注入上下文；模型会在需要时通过工具查询或更新记忆：

- 用户明确说“记住……”时，模型应调用 `remember`。
- 用户问“你记得什么……”或问题依赖历史偏好时，模型可调用 `recall_memory`。
- 用户要求忘记某条记忆时，模型应先查询 id，再调用 `forget_memory`。
- 密码、API key、访问令牌等敏感秘密不应写入长期记忆。

记忆文件是普通 JSON 数组。每条记忆包含 `id`、`category`、`content`、`created_at` 和 `updated_at` 字段。

## 模型配置

模型配置在 `config.json` 中：

```json
{
  "base_url": "https://api.deepseek.com",
  "api_key": "ollama",
  "model": "deepseek-v4-flash",
  "stream": true
}
```

字段说明：

- `base_url`：OpenAI 兼容接口地址；为空字符串或 `null` 时使用 OpenAI SDK 默认官方地址。
- `api_key`：接口密钥。
- `model`：模型名称。
- `stream`：是否使用 LangGraph `messages` stream mode 流式输出模型可见文本。

如果 `config.json` 不存在、JSON 格式错误，或读取失败，程序会自动使用默认 Ollama 配置：

```json
{
  "base_url": "http://127.0.0.1:11434/v1",
  "api_key": "ollama",
  "model": "qwen3:8b",
  "stream": false
}
```

## 使用本地模型 API

本地 Ollama 或其它 OpenAI 兼容服务可以这样配置：

```json
{
  "base_url": "http://10.6.22.1:11434/v1",
  "api_key": "ollama",
  "model": "qwen3:8b",
  "stream": false
}
```

## 改用 OpenAI 官方 API

本项目通过 `langchain-openai` 调用 OpenAI 兼容接口。要从本地 OpenAI 兼容接口切到 OpenAI 官方 API，把 `config.json` 改成：

```json
{
  "base_url": "",
  "api_key": "你的 OpenAI API key",
  "model": "你要使用的官方模型名",
  "stream": true
}
```

当 `base_url` 为空字符串或 `null` 时，`ChatOpenAI` 会使用默认 OpenAI 官方 API 地址；当它有值时，会使用指定的兼容接口地址。其它兼容 OpenAI 协议的服务，例如 DeepSeek，也可以通过填写对应的 `base_url`、`api_key` 和 `model` 使用。
