# Github search-report Agent

这是一个基于 LangGraph + LangChain 的最小可扩展 Agent 项目。项目按 Agent、Runner、Tools、Handoff、Guardrail、Session、Tracing、Result 分层；每个具体 Agent 文件负责定义自己的 `StateGraph` 流程，`runner/` 负责调用对应 graph 并管理执行循环，`run.py` 只负责终端入口调度，终端打印放在 `result/`。`tools/` 目录负责放置具体工具函数和工具注册适配。新增工具时，只需要定义函数、写好 docstring，然后加入 `tools/__init__.py` 的 `TOOLS` 数组。

## 文件结构

```text
agent/
|-- agent/                  # Agent 描述能力：名称、指令、工具、协作钩子
|-- config.json             # 模型配置：base_url、api_key、model、stream
|-- data/
|   `-- memory.json         # 本地长期记忆文件，默认被 .gitignore 忽略
|-- guardrail/              # Guardrail：安全和质量检查接口
|-- handoff/                # Handoff：多 Agent 协作路由接口
|-- requirements.txt        # LangGraph / LangChain 依赖
|-- result/                 # Result：结构化结果、流事件和终端打印
|-- run.py                  # 终端入口：读取配置并调用 runner/result
|-- runner/                 # Runner：配置、LangGraph 构建、执行循环
|-- session/                # Session：记忆和会话状态接口
|-- tools/
|   |-- __init__.py         # 具体工具注册表：TOOLS 和 TOOL_MAP
|   |-- code_tools.py       # 文件结构、读文件、创建文件、编辑文件、编译检查
|   |-- math_tools.py       # 数学工具
|   |-- memory_tools.py     # 长期记忆工具
|   |-- registry.py         # 工具注册表对象和 LangChain 适配
|   |-- security_tools.py   # 随机密钥工具
|   `-- web_tools.py        # 网络搜索和网页正文抓取工具
|-- tracing/                # Tracing：轻量调试事件
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

`runner.run_agent(user_input, max_tool_calls=10)` 会把用户输入放入 LangGraph 状态，然后在 `agent -> tools -> agent` 图中循环。模型通过 LangChain 的 `bind_tools()` 选择工具；工具执行结果会以 `ToolMessage` 返回给图，直到模型给出最终回答。

当 `stream=true` 时，`runner.stream_agent_events()` 使用 LangGraph 的 `messages` + `updates` stream mode 产出每一次 `agent` 模型节点产生的可见文本事件；`result/` 负责把这些事件打印到终端。如果模型先输出文字再调用工具，这段文字也会显示，不只显示最终回答。

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

`tools/registry.py` 会把 `TOOLS` 中的普通 Python 函数包装成 LangChain `StructuredTool`，`runner/` 通过 `ChatOpenAI(...).bind_tools(...)` 交给模型。`tools/__init__.py` 维护具体工具函数的注册表。

## 架构分层

- `agent/`：描述 Agent 的名称、指令、工具、handoff、guardrail、session 和 tracing 能力。
- `runner/`：调用各 Agent 自己的 LangGraph，并管理执行循环和流式事件。
- `tools/`：保存具体工具函数，并把它们适配成 LangChain 工具。
- `handoff/`：预留多 Agent 协作路由接口。
- `guardrail/`：预留输入、输出或工具调用的安全和质量检查接口。
- `session/`：封装长期记忆的读取和写入接口。
- `tracing/`：记录轻量调试事件。
- `result/`：定义 `RunResult` 和 `StreamEvent` 等结构化返回类型，并集中处理终端打印。

## Agent Graph 定制

默认情况下，`runner.create_agent_graph()` 会读取 `Agent.graph_factory`。每个具体 Agent 文件都定义自己的 graph factory，并在该文件中直接声明 `StateGraph`、node、edge 和条件路由；未来要改某个 Agent 的执行流程，只需要改对应文件。

例如 [agent/github_search_agent.py](agent/github_search_agent.py) 中：

```python
def create_github_search_graph(**kwargs):
    deps = import_langgraph_dependencies()
    StateGraph = deps["StateGraph"]
    START = deps["START"]
    END = deps["END"]

    def call_model(state):
        ...

    def call_tools(state):
        ...

    def should_continue(state):
        ...

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", call_tools)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
    graph_builder.add_edge("tools", "agent")
    return graph_builder.compile(), HumanMessage
```

然后在创建 Agent 时绑定：

```python
return Agent(
    name=AGENT_NAME,
    instructions=INSTRUCTIONS,
    graph_factory=create_github_search_graph,
)
```

当前约定：

- `RequirementAgent`：`create_requirement_graph`，默认 model-only。
- `GitHubSearchAgent`：`create_github_search_graph`，默认 ReAct tools graph。
- `RepoAnalysisAgent`：`create_repo_analysis_graph`，默认 ReAct tools graph。
- `ScoringAgent`：`create_scoring_graph`，默认 model-only。
- `ReportAgent`：`create_report_graph`，默认 model-only。
- `CriticAgent_*`：`create_critic_graph`，默认 ReAct tools graph，保留 `web_search` 核查能力。

`graph_factory` 接收 `config`、`max_tool_calls`、`agent`、`on_tool_start`、`on_tool_end` 参数，并返回 `(graph, HumanMessage)`。

## 单 Agent 调试

`runner.run_single_agent_debug()` 可用于单独测试某个 Agent。输入可以由自定义 mock 函数生成 JSON；也可以传入 `mock_output_factory` 跳过真实 LLM 调用，只测试 OutputGuardrail 或 Critic 流程。

直接运行默认 mock 调试：

```bash
python test.py
```

指定 Agent 并启用 OutputGuardrail：

```bash
python test.py --agent RepoAnalysisAgent --guardrail
```

跳过 mock output、调用真实模型图：

```bash
python test.py --agent GitHubSearchAgent --real-agent
```

接入 CriticAgent 做语义审查：

```bash
python test.py --agent GitHubSearchAgent --critic
```

调试链路支持：

- 只跑单个 Agent
- 使用 mock JSON 输入
- 使用 mock output 跳过真实模型
- 可选接 OutputGuardrail
- 可选接 CriticAgent 做语义审查

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
- `stream`：是否使用 LangGraph 流式输出模型可见文本。开启后会输出所有 `agent` 模型节点返回的文本，不只输出最终答案。

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
