from typing import Callable, Literal

from agent import Agent, create_default_agent
from result import RunResult, StreamEvent
from tracing import Tracer

from .config import load_config
from .langgraph_dependencies import import_langgraph_dependencies
from .messages import final_message_content, message_content_text, message_id

ToolCallback = Callable[[str], None]


def build_model(ChatOpenAI, langchain_tools, config: dict):
    return build_chat_model(ChatOpenAI, config).bind_tools(langchain_tools)


def build_chat_model(ChatOpenAI, config: dict):
    model_options = {
        "model": config["model"],
        "api_key": config["api_key"],
        "temperature": 0,
    }
    if config["base_url"]:
        model_options["base_url"] = config["base_url"]

    return ChatOpenAI(**model_options)


def create_agent_graph(
    config: dict,
    max_tool_calls: int = 10,
    agent: Agent | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
):
    agent = agent or create_default_agent()
    if agent.graph_factory is not None:
        return agent.graph_factory(
            config=config,
            max_tool_calls=max_tool_calls,
            agent=agent,
            on_tool_start=on_tool_start,
            on_tool_end=on_tool_end,
        )

    return create_react_agent_graph(
        config=config,
        max_tool_calls=max_tool_calls,
        agent=agent,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )


def create_react_agent_graph(
    config: dict,
    max_tool_calls: int = 10,
    agent: Agent | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
):
    agent = agent or create_default_agent()
    deps = import_langgraph_dependencies()
    HumanMessage = deps["HumanMessage"]
    SystemMessage = deps["SystemMessage"]
    ToolMessage = deps["ToolMessage"]
    ChatOpenAI = deps["ChatOpenAI"]
    END = deps["END"]
    START = deps["START"]
    StateGraph = deps["StateGraph"]
    add_messages = deps["add_messages"]
    Annotated = deps["Annotated"]
    TypedDict = deps["TypedDict"]

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        llm_calls: int

    langchain_tools = agent.tools.as_langchain_tools(deps["StructuredTool"])
    langchain_tool_map = {tool.name: tool for tool in langchain_tools}
    model_with_tools = build_model(ChatOpenAI, langchain_tools, config)
    tracer = agent.tracer or Tracer()

    def call_model(state: AgentState):
        tracer.record("model_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model_with_tools.invoke(messages)
        tracer.record("model_end", has_tool_calls=bool(response.tool_calls))
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    def call_tools(state: AgentState):
        last_message = state["messages"][-1]
        tool_messages = []

        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool = langchain_tool_map.get(tool_name)
            tracer.record("tool_start", tool_name=tool_name)

            if tool is None:
                tool_result = f"模型请求了未知工具：{tool_name}"
            else:
                if on_tool_start:
                    on_tool_start(tool_name)

                try:
                    tool_result = str(tool.invoke(tool_call["args"]))
                except Exception as exc:
                    tool_result = f"工具执行失败：{exc}"
                finally:
                    if on_tool_end:
                        on_tool_end(tool_name)

            tracer.record("tool_end", tool_name=tool_name)
            tool_messages.append(
                ToolMessage(
                    content=tool_result,
                    name=tool_name,
                    tool_call_id=tool_call["id"],
                )
            )

        return {"messages": tool_messages}

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        last_message = state["messages"][-1]
        if last_message.tool_calls and state.get("llm_calls", 0) <= max_tool_calls:
            return "tools"

        return END

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_node("tools", call_tools)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges("agent", should_continue, ["tools", END])
    graph_builder.add_edge("tools", "agent")

    return graph_builder.compile(), HumanMessage


def create_model_only_agent_graph(
    config: dict,
    max_tool_calls: int = 10,
    agent: Agent | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
):
    agent = agent or create_default_agent()
    deps = import_langgraph_dependencies()
    HumanMessage = deps["HumanMessage"]
    SystemMessage = deps["SystemMessage"]
    ChatOpenAI = deps["ChatOpenAI"]
    END = deps["END"]
    START = deps["START"]
    StateGraph = deps["StateGraph"]
    add_messages = deps["add_messages"]
    Annotated = deps["Annotated"]
    TypedDict = deps["TypedDict"]

    class AgentState(TypedDict):
        messages: Annotated[list, add_messages]
        llm_calls: int

    model = build_chat_model(ChatOpenAI, config)
    tracer = agent.tracer or Tracer()

    def call_model(state: AgentState):
        tracer.record("model_only_start", llm_calls=state.get("llm_calls", 0))
        messages = [
            SystemMessage(content=agent.system_prompt()),
            *state["messages"],
        ]
        response = model.invoke(messages)
        tracer.record("model_only_end")
        return {
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    graph_builder = StateGraph(AgentState)
    graph_builder.add_node("agent", call_model)
    graph_builder.add_edge(START, "agent")
    graph_builder.add_edge("agent", END)

    return graph_builder.compile(), HumanMessage


def build_initial_state(HumanMessage, user_input: str) -> dict:
    return {
        "messages": [HumanMessage(content=user_input)],
        "llm_calls": 0,
    }


def check_guardrails(agent: Agent, user_input: str) -> str | None:
    for guardrail in agent.guardrails:
        result = guardrail.run(user_input)
        if not result.ok:
            return result.message or f"Guardrail blocked input: {guardrail.name}"

    return None


def choose_handoff(agent: Agent, user_input: str):
    for handoff in agent.handoffs:
        decision = handoff.decide(user_input)
        if decision.should_handoff:
            return decision

    return None


def stream_agent_events(
    user_input: str,
    max_tool_calls: int = 10,
    config: dict | None = None,
    agent: Agent | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
):
    config = config or load_config()
    agent = agent or create_default_agent()
    blocked_message = check_guardrails(agent, user_input)
    if blocked_message:
        yield StreamEvent(type="text", content=blocked_message)
        return

    graph, HumanMessage = create_agent_graph(
        config,
        max_tool_calls,
        agent=agent,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )
    initial_state = build_initial_state(HumanMessage, user_input)

    streamed_text = ""
    streamed_message_ids = set()
    streamed_text_by_id = {}
    current_stream_message_id = None

    for part in graph.stream(
        initial_state,
        stream_mode=["messages", "updates"],
        version="v2",
    ):
        part_type = part["type"]

        if part_type == "messages":
            message_chunk, metadata = part["data"]
            if metadata.get("langgraph_node") != "agent":
                continue

            content = message_content_text(message_chunk)
            if not content:
                continue

            chunk_id = message_id(message_chunk)
            if chunk_id and current_stream_message_id and chunk_id != current_stream_message_id:
                yield StreamEvent(type="line_break")

            yield StreamEvent(type="text", content=content)
            streamed_text += content

            if chunk_id:
                current_stream_message_id = chunk_id
                streamed_message_ids.add(chunk_id)
                streamed_text_by_id[chunk_id] = (
                    streamed_text_by_id.get(chunk_id, "") + content
                )

        elif part_type == "updates":
            agent_update = part["data"].get("agent")
            if not agent_update:
                continue

            for message in agent_update.get("messages", []):
                content = message_content_text(message)
                if not content:
                    continue

                current_message_id = message_id(message)
                if current_message_id in streamed_message_ids:
                    continue

                if current_message_id and streamed_text_by_id.get(current_message_id) == content:
                    continue

                if not current_message_id and content in streamed_text:
                    continue

                yield StreamEvent(type="line_break")
                yield StreamEvent(type="text", content=content)
                streamed_text += content

                if current_message_id:
                    streamed_message_ids.add(current_message_id)


def run_agent(
    user_input: str,
    max_tool_calls: int = 10,
    config: dict | None = None,
    agent: Agent | None = None,
    on_tool_start: ToolCallback | None = None,
    on_tool_end: ToolCallback | None = None,
) -> RunResult:
    config = config or load_config()
    agent = agent or create_default_agent()
    blocked_message = check_guardrails(agent, user_input)
    if blocked_message:
        return RunResult(
            final_output=blocked_message,
            metadata={"blocked_by_guardrail": True},
        )

    handoff_decision = choose_handoff(agent, user_input)
    graph, HumanMessage = create_agent_graph(
        config,
        max_tool_calls,
        agent=agent,
        on_tool_start=on_tool_start,
        on_tool_end=on_tool_end,
    )
    initial_state = build_initial_state(HumanMessage, user_input)
    final_state = graph.invoke(initial_state)
    return RunResult(
        final_output=final_message_content(final_state["messages"]),
        metadata={
            "handoff": handoff_decision.target_agent if handoff_decision else None,
            "message_count": len(final_state["messages"]),
        },
    )
