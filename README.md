# Simple Tool Agent

这是一个最小可扩展 Agent 项目。`run.py` 负责模型调用和工具循环，`tools/` 目录负责放置工具函数。新增工具时，只需要定义函数、写好 docstring，然后加入 `tools/__init__.py` 的 `TOOLS` 数组。

## 文件结构

```text
agent/
|-- run.py                  # 项目入口：调用模型、解析工具 JSON、多轮执行工具
|-- test.py                 # 兼容入口：直接复用 run_agent
|-- tools/
|   |-- __init__.py         # 工具注册表：TOOLS 和 TOOL_MAP
|   |-- code_tools.py       # 文件结构、读文件、创建文件、编辑文件、编译检查
|   |-- math_tools.py       # 数学工具
|   |-- security_tools.py   # 随机密钥工具
|   `-- web_tools.py        # 网络搜索和网页正文抓取工具
`-- README.md
```

## 运行

先安装 OpenAI Python SDK：

```bash
pip install openai
```

然后运行：

```bash
python run.py
```

`run_agent(user_input, max_tool_calls=8)` 会循环调用模型。模型如果返回普通文本，就直接结束；如果返回工具调用 JSON，就执行工具并把结果交还给模型，直到模型认为信息足够并给出最终回答。

工具调用 JSON 格式：

```json
{"tool": "工具名", "arguments": {"参数名": "参数值"}}
```

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

`run.py` 会自动读取函数签名和 docstring，生成 system prompt，并自动构造工具分发映射。

## 使用本地模型 API

当前默认配置适合 Ollama 这类 OpenAI 兼容本地接口：

```python
DEFAULT_BASE_URL = "http://10.6.22.1:11434/v1"
API_KEY = "ollama"
MODEL = "qwen3:8b"
```

也可以用环境变量覆盖：

```bash
export OPENAI_BASE_URL="http://10.6.22.1:11434/v1"
export OPENAI_API_KEY="ollama"
export OPENAI_MODEL="qwen3:8b"
python run.py
```

## 改用 OpenAI 官方 API

本项目使用 OpenAI Python SDK。要从本地 OpenAI 兼容接口切到 OpenAI 官方 API，需要让 SDK 使用默认官方 base URL，并提供真实 API key。如果有其它的API比如说Deepseek,也可以使用其openai接口，具体查看Deepseek官方文档。

Linux/macOS：

```bash
export OPENAI_API_KEY="你的 OpenAI API key"
export OPENAI_MODEL="你要使用的官方模型名"
export OPENAI_BASE_URL=
python run.py
```

PowerShell：

```powershell
$env:OPENAI_API_KEY = "你的 OpenAI API key"
$env:OPENAI_MODEL = "你要使用的官方模型名"
$env:OPENAI_BASE_URL = ""
python run.py
```

也可以直接修改 `run.py`：

```python
BASE_URL = ""
API_KEY = "你的 OpenAI API key"
MODEL = "你要使用的官方模型名"
```

当 `OPENAI_BASE_URL` 或 `BASE_URL` 为空字符串时，`OpenAI()` 会使用 SDK 默认的 OpenAI 官方 API 地址；当它有值时，会使用指定的兼容接口地址。

