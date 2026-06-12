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
|   |-- github_search_*.py  # GitHubSearchAgent DAG 节点（查询/摄入/检索/精排/过滤）
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

例如 [agent/github_search_agent.py](agent/github_search_agent.py) 中，GitHubSearchAgent 使用 DAG 流水线模式（非 agent-loop），含内部自审：

```python
def _create_dag_graph(**kwargs):
    # 6 节点线性流水线 + 内部自审
    builder = StateGraph(AgentState)
    builder.add_node("convert_query", _convert_query)
    builder.add_node("ingest_github_repos", _ingest_github_repos)
    builder.add_node("dense_retrieval", _dense_retrieval)
    builder.add_node("llm_reranking", _llm_reranking)
    builder.add_node("threshold_filtering", _threshold_filtering)
    builder.add_node("self_review", _self_review)
    builder.add_node("finalize", _finalize)

    builder.add_edge(START, "convert_query")
    builder.add_edge("convert_query", "ingest_github_repos")
    builder.add_edge("ingest_github_repos", "dense_retrieval")
    builder.add_edge("dense_retrieval", "llm_reranking")
    builder.add_edge("llm_reranking", "threshold_filtering")
    builder.add_edge("threshold_filtering", "self_review")
    builder.add_edge("self_review", "finalize")
    builder.add_edge("finalize", END)
    return builder.compile(), HumanMessage
```

流水线流程（6 节点 + 最终输出）：

1. **convert_query** — 结构化 JSON（project_type + keywords + language） → 冒号分隔的搜索关键词
2. **ingest_github_repos** — 逐关键词调用 GitHub Search API，并发抓取每个仓库的 README + docs
3. **dense_retrieval** — SentenceTransformer + BM25 混合语义检索
4. **llm_reranking** — DeepSeek 单次调用对 top-N 精排打分
5. **threshold_filtering** — 按 stars + 精排分阈值过滤低质量仓库
6. **self_review** — 内部逐条审核相关性：硬规则预检（语言匹配）+ LLM 语义审核，不通过的跳过并用后续候选补位
7. **finalize** — 截取 top_n 并输出 JSON

节点函数定义在 `tools/github_search_*.py` 中。

> **注意：** GitHubSearchAgent 使用 DAG 内部自审，不走外部 CriticAgent 审查流水线。其他 Agent 仍使用默认 ReAct agent-loop 模式（agent ↔ tools 循环）。

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
  "api_key": "你的 API key",
  "model": "deepseek-v4-pro",
  "stream": true,

  "github_api_key": "github_pat_xxx",
  "github_max_results": 100,
  "github_per_page": 25,
  "dense_retrieval_k": 100,
  "llm_rerank_top_n": 50,
  "retrieval_alpha": 0.7,
  "min_stars": 50,
  "rerank_threshold": 5.5
}
```

### 基础字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `base_url` | string | OpenAI 兼容接口地址；为空时使用默认官方地址 |
| `api_key` | string | LLM API 密钥 |
| `model` | string | 模型名称 |
| `stream` | bool | 是否使用流式输出 |

### GitHubSearchAgent DAG 专用字段

这些字段控制 GitHubSearchAgent 的 DAG 流水线参数：

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `github_api_key` | `""` | GitHub Personal Access Token。留空时尝试读环境变量 `GITHUB_API_KEY`。需要 `public_repo` 权限。创建地址：https://github.com/settings/tokens |
| `github_max_results` | `100` | 每个关键词从 GitHub API 拉取的最大仓库数 |
| `github_per_page` | `25` | GitHub API 每页返回条数（max 100） |
| `dense_retrieval_k` | `100` | 语义检索（SentenceTransformer + BM25）输出的候选数 |
| `llm_rerank_top_n` | `50` | 送入 LLM 精排的 top-N 候选数 |
| `retrieval_alpha` | `0.7` | 混合检索权重：α × Dense + (1-α) × BM25。0=纯 BM25，1=纯语义 |
| `min_stars` | `50` | 星数阈值：低于此值**且**精排分低于 `rerank_threshold` 的仓库被丢弃 |
| `rerank_threshold` | `5.5` | 精排分阈值：配合 `min_stars` 共同决定过滤 |

如果 `config.json` 不存在、JSON 格式错误，或读取失败，程序会自动使用默认配置。

## 下载语义检索模型

GitHubSearchAgent 使用 `sentence-transformers/all-mpnet-base-v2` 做语义向量检索。模型约 1.7GB，需要离线下载到项目根目录。

```bash
# 安装 huggingface CLI（新版用 hf 命令）
pip install huggingface_hub

# 国内用户建议先设镜像
export HF_ENDPOINT=https://hf-mirror.com

# 下载到项目根目录
hf download sentence-transformers/all-mpnet-base-v2 --local-dir ./all-mpnet-base-v2
```

下载完成后目录结构：

```text
all-mpnet-base-v2/
├── config.json
├── model.safetensors
├── tokenizer.json
├── vocab.txt
└── ...
```

程序会自动检测 `./all-mpnet-base-v2/` 是否存在并优先使用本地模型，无需额外配置。

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

## D：评分、Guardrail 与质量实现

D 模块已经实现确定性评分、规则检查、Critic 语义检查和结构化 Result。

### 评分状态机

`ScoringAgent` 的 `graph_factory` 使用独立状态机：

```text
START
  -> parse_input
  -> score_projects
  -> validate_scores
  -> finalize
  -> END
```

评分不依赖 LLM 自由打分，固定使用 100 分制：

| 维度 | 满分 |
|---|---:|
| 功能匹配度 `function_match` | 30 |
| 部署便利性 `deployment` | 20 |
| 二次开发友好度 `developer_friendliness` | 20 |
| 社区活跃度 `community_activity` | 15 |
| 文档完善度 `documentation` | 10 |
| 许可证友好度 `license_friendliness` | 5 |

输出字段为 `project`、`total_score`、`scores`、`reason`、`evidence`、`weights` 和 `rank`。

### 评分与质量工具

`tools/scoring_tools.py` 提供：

- `score_projects`：合并仓库分析与 GitHub metadata，计算六维评分并排序。
- `validate_project_result`：检查空候选、README、license、评分字段、范围、总分复算、排序和疑似幻觉字段。
- `validate_report`：检查报告九个章节、Markdown 表格、参考来源和未知项目名。

### 双层质量检查

每个阶段先运行 `OutputGuardrail` 做程序规则检查；规则通过后再调用对应的 `CriticAgent` 做语义检查。Scoring 阶段已加入主流水线：

```text
ScoringAgent
  -> OutputGuardrail（字段、范围、总分、排序）
  -> CriticAgent_ScoringAgent（证据、约束、一致性）
  -> ReportAgent
```

规则检查失败不会调用 Critic，而是携带失败原因重试当前 Agent。Critic 检查评分依据、缺失信息是否被不合理高分、硬约束是否正确降权。

### Result schema

`RunResult.metadata` 始终包含：

```python
{
    "requirements": {},
    "projects": [],
    "scores": [],
    "guardrail_warnings": [],
    "agent_steps": [],
}
```

### 单 Agent 调试

```bash
python test.py --agent ScoringAgent --guardrail
python test.py --agent ScoringAgent --real-agent --guardrail
python test.py --agent ScoringAgent --critic
python test.py --agent ReportAgent --guardrail
```

`--real-agent` 会实际运行 ScoringAgent 的状态机，但该状态机本身不调用 LLM。`--critic` 会在规则检查配合之外调用真实 Critic LLM。

### 自动测试

```bash
python -B -m unittest discover -s tests -v
```

完整检查结果见 [QUALITY_REPORT.md](QUALITY_REPORT.md)。
