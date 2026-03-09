# Async streaming response handler for LLM interactions
import uuid
from typing import AsyncIterator, Awaitable, Callable, cast

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from src.graph.graph_builder import get_graph_cached


def new_run_id() -> str:
    return str(uuid.uuid4())


async def stream_response(prompt: str) -> AsyncIterator[str]:
    """Stream assistant response text."""
    run_id = new_run_id()

    input_state = {
        "prompt": prompt,
        "messages": [HumanMessage(content=prompt, input_type="user_input")],
        "last_recommendations": st.session_state.last_recommendations,
        "last_checked_books": st.session_state.last_checked_books,
    }

    stream = get_graph_cached().astream(
        input=input_state,
        config={
            "configurable": {"thread_id": st.session_state.thread_id},
            "metadata": {"run_id": run_id},
        },
        stream_mode=["messages", "updates"],
        subgraphs=True,
    )
    latest_recommendations = st.session_state.last_recommendations
    latest_checked_books = st.session_state.last_checked_books
    try:
        async for namespace, stream_mode, data in stream:
            # Logic for logging debug messages
            if stream_mode == "messages":
                chunk, metadata = data
                if not isinstance(metadata, dict):
                    continue
                node = metadata.get("langgraph_node", "unknown")
                valid_nodes = {"agent", "input_security"}
                # Stream AIMessage content character by character
                chunk_content = getattr(chunk, "content", None)
                if (
                    chunk_content
                    and node in valid_nodes
                    and isinstance(chunk, AIMessage)
                ):
                    for char in chunk_content:
                        yield char
            if stream_mode == "updates":
                updates = []
                if isinstance(data, dict):
                    updates.append(data)
                    for value in data.values():
                        if isinstance(value, dict):
                            updates.append(value)
                for update in updates:
                    if "last_recommendations" in update:
                        latest_recommendations = update.get("last_recommendations")
                    if "last_checked_books" in update:
                        latest_checked_books = update.get("last_checked_books")
    except Exception:
        error_message = (
            "Oops! I hit an error while generating your book recommendations. "
            "Please try again, or let me know if there's another way I can help you discover great reads!\n"
        )
        for char in error_message:
            yield char
    finally:
        if latest_recommendations is not None:
            st.session_state.last_recommendations = latest_recommendations
        if latest_checked_books is not None:
            st.session_state.last_checked_books = latest_checked_books
        aclose = getattr(stream, "aclose", None)
        if callable(aclose):
            await cast(Callable[[], Awaitable[object]], aclose)()
