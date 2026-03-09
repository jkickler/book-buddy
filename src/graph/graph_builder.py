# Graph builder for LLM agent workflow with security, summarization, and tool orchestration
import json

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from loguru import logger

from src.graph.prompts import (
    agent_system_prompt,
    create_secure_prompt,
    summarization_prompt,
)
from src.graph.schemas import GraphState
from src.graph.tools.enrich_and_score import enrich_and_score_tool
from src.graph.tools.query_to_read_list import query_to_read_list_tool
from src.graph.tools.recommend_by_profile import recommend_by_profile_tool
from src.graph.tools.save_to_read_list import save_to_read_list_tool
from src.security.prompt_injection import OutputValidator, PromptInjectionFilter

tools = [
    enrich_and_score_tool,
    recommend_by_profile_tool,
    save_to_read_list_tool,
    query_to_read_list_tool,
]

MODEL = ChatOpenAI(model="gpt-5.2", streaming=False)
MODEL_WITH_TOOLS = MODEL.bind_tools(tools)
SUMMARIZ_MODEL = ChatOpenAI(model="gpt-5-mini", streaming=False)

SUMMARIZ_THRESHOLD = 10
MESSAGES_TO_KEEP = 4


async def summarization_node(state: GraphState) -> dict:
    """Summarizes older messages when the threshold is exceeded to manage conversation length.

    - Checks if messages exceed 10 (threshold).
    - Keeps the 4 most recent messages.
    - Summarizes older messages using a model.
    - Updates state with new summary and removes summarized messages.
    """
    messages = state.get("messages") or []
    existing_summary = state.get("summary", "")

    msg_count = len(messages)
    if msg_count <= SUMMARIZ_THRESHOLD:
        return {}

    messages_to_summarize = messages[:-MESSAGES_TO_KEEP]

    summary_content = (
        f"Previous conversation summary:\n{existing_summary}\n\n"
        if existing_summary
        else ""
    )
    summary_content += "Recent messages to summarize:\n"
    for msg in messages_to_summarize:
        if isinstance(msg, HumanMessage):
            summary_content += f"User: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            summary_content += f"Assistant: {msg.content}\n"

    response = await SUMMARIZ_MODEL.ainvoke(
        [
            SystemMessage(content=summarization_prompt),
            HumanMessage(content=summary_content),
        ]
    )

    new_summary = response.content
    removals = [RemoveMessage(id=msg.id) for msg in messages_to_summarize if msg.id]
    sys_msg = [SystemMessage(content=f"[Conversation Summary]\n{new_summary}")]

    return {
        "messages": removals + sys_msg,
        "summary": new_summary,
    }


async def agent_node(state: GraphState) -> dict:
    """LLM decides whether to call tools or respond. Output is validated for sensitive data leakage.
    If sensitive data is detected, it is filtered out."""
    messages = state.get("messages") or []
    logger.info("AGENT: START")

    # Sanitize messages by collecting tool_call_ids from AIMessages
    # and filtering ToolMessages to only include those with matching ids
    sanitized_messages = []
    tool_call_ids = set()
    for msg in messages:
        if isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tool_call in tool_calls:
                tool_call_id = (
                    tool_call.get("id") if isinstance(tool_call, dict) else None
                )
                if tool_call_id:
                    tool_call_ids.add(tool_call_id)
            sanitized_messages.append(msg)
            continue

        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            if tool_call_id in tool_call_ids:
                sanitized_messages.append(msg)
            continue

        sanitized_messages.append(msg)

    system_message = SystemMessage(content=agent_system_prompt)
    response = await MODEL_WITH_TOOLS.ainvoke([system_message, *sanitized_messages])

    # Log decision
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc.get("name", "unknown") for tc in response.tool_calls]
        logger.info(f"AGENT DECISION: Call tools - {tool_names}")
    else:
        logger.info("AGENT DECISION: Respond directly")

    content = response.content
    if isinstance(content, list):
        content = "".join(
            str(c) if isinstance(c, str) else c.get("text", "") for c in content
        )

    validator = OutputValidator()
    validated_content = validator.filter_response(content)
    if validated_content != content:
        logger.warning("AGENT: Output was filtered by security validator")
        response = AIMessage(content=validated_content)

    return {"messages": [response]}


async def extract_tool_state_node(state: GraphState) -> dict:
    """Extract state from the latest ToolMessage when applicable."""
    messages = state.get("messages") or []
    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = msg.content
            if isinstance(content, list):
                content = "".join(
                    str(c) if isinstance(c, str) else c.get("text", "") for c in content
                )
            data = json.loads(str(content))
        except json.JSONDecodeError:
            return {}

        if msg.name == "recommend_by_profile":
            if data.get("status") == "ok":
                return {"last_recommendations": data.get("candidates", [])}
            return {}

        if msg.name == "enrich_and_score":
            if data.get("status") != "ok":
                return {}
            results = data.get("results", [])
            checked = []
            for item in results:
                if item.get("status") == "ok" and item.get("enriched_book"):
                    checked.append(
                        {
                            "enriched_book": item.get("enriched_book"),
                            "similarity_scores": item.get("similarity_scores"),
                        }
                    )
            if checked:
                return {"last_checked_books": checked}
            return {}

        return {}

    return {}


def should_continue(state: GraphState) -> str:
    """Decides whether to continue with tools or end the conversation."""
    messages = state.get("messages") or []
    last = messages[-1]
    tool_calls = getattr(last, "tool_calls", None)
    if tool_calls:
        tool_names = [tc.get("name", "unknown") for tc in tool_calls]
        logger.info(f"ROUTING: Continue - {tool_names}")
        return "tools"
    logger.info("ROUTING: End")
    return END


async def input_security_node(state: GraphState) -> dict:
    """Validates user input for prompt injection attempts and applies delimiter security."""
    messages = state.get("messages") or []
    if not messages:
        return {"is_blocked": False}

    last_message = messages[-1]
    if not isinstance(last_message, HumanMessage):
        return {"is_blocked": False}

    content = last_message.content
    if isinstance(content, list):
        content = "".join(
            str(c) if isinstance(c, str) else c.get("text", "") for c in content
        )
    else:
        content = str(content)

    # Check for injection patterns first
    filter_instance = PromptInjectionFilter()
    if filter_instance.detect_injection(content):
        logger.warning("SECURITY: BLOCKED")
        return {
            "is_blocked": True,
            "messages": [
                AIMessage(
                    content=(
                        "I'm sorry, but I couldn't process your request because it contains "
                        "content that appears to be attempting to manipulate my behavior. "
                        "Please rephrase your question and try again."
                    )
                )
            ],
        }

    # Wrap user input in security delimiters
    logger.info("SECURITY: PASSED")
    secured_content = create_secure_prompt(content)
    secured_messages = messages[:-1] + [HumanMessage(content=secured_content)]

    return {"is_blocked": False, "messages": secured_messages}


def route_after_input_security(state: GraphState) -> str:
    """Routes to END if blocked, otherwise to summarize."""
    is_blocked = state.get("is_blocked")
    if is_blocked:
        logger.warning("ROUTING: Security blocked")
        return END
    logger.info("ROUTING: Continue to summarize")
    return "summarize"


tool_node = ToolNode(tools)
builder = StateGraph(GraphState)

builder.add_node("input_security", input_security_node)
builder.add_node("summarize", summarization_node)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("extract_tool_state", extract_tool_state_node)

builder.add_edge(START, "input_security")
builder.add_conditional_edges(
    "input_security", route_after_input_security, ["summarize", END]
)
builder.add_edge("summarize", "agent")
builder.add_conditional_edges("agent", should_continue, ["tools", END])
builder.add_edge("tools", "extract_tool_state")
builder.add_edge("extract_tool_state", "agent")

memory = InMemorySaver()
graph = builder.compile(checkpointer=memory)


def get_graph_cached():
    """Return the compiled graph cached in session state."""
    if "compiled_graph" not in st.session_state:
        st.session_state.compiled_graph = builder.compile(checkpointer=memory)
    return st.session_state.compiled_graph
