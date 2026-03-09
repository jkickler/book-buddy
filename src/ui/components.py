# Streamlit UI components including sidebar, chat history, and vector store setup
import time
import uuid

import streamlit as st

from ..vectorstore.chroma import (
    DEFAULT_PERSIST_DIRECTORY,
    load_book_vector_store,
    vector_store_exists,
)
from ..vectorstore.ingest import ingest_csv

TITLE_HTML = """
<div style='text-align: center; padding: 30px 0;'>
    <h1 style='margin-bottom: 10px;'>📚 BOOK BUDDY 📚</h1>
    <p style='color: #666; font-size: 18px;'>
        Get personalized book recommendations based on what you've read.
        <br>Just share a book you enjoyed and I'll find more like it!
    </p>
</div>
"""

USER_ICON = "👤"
ASSISTANT_ICON = "🧠"

STATUS_INDICATOR_HTML = """
<div style='display: flex; align-items: center; padding: 10px;'>
    <div style='display: inline-block; width: 12px; height: 12px; background: #1f77b4; border-radius: 50%; animation: pulse 1.5s infinite; margin-right: 10px;'></div>
    <span style='color: #666; font-weight: 500;'>{text}</span>
</div>
<style>
@keyframes pulse {{
    0% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.5; transform: scale(1.1); }}
    100% {{ opacity: 1; transform: scale(1); }}
}}
</style>
"""

CUSTOM_CSS = """
<style>
    /* Chat message styling */
    div[data-testid="chat-message-user"] {
        border-radius: 12px 12px 0 12px !important;
        padding: 12px 16px !important;
        margin-left: 40px !important;
    }
    div[data-testid="chat-message-assistant"] {
        border-radius: 12px 12px 12px 0 !important;
        padding: 12px 16px !important;
        margin-right: 40px !important;
    }
    
    /* Input field styling */
    div[data-testid="stChatInput"] {
        border-radius: 12px !important;
    }

</style>
"""


def get_status_indicator(text: str = "Thinking...") -> str:
    """Return HTML for a pulsing status indicator."""
    return STATUS_INDICATOR_HTML.format(text=text)


def handle_refresh() -> None:
    """Clear chat-related session state to start fresh conversation."""
    st.session_state.chat_msgs = []
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()


def render_sidebar() -> None:
    """Render the sidebar with refresh button and user guide."""
    with st.sidebar:
        st.markdown("# 📖 USER GUIDE", text_alignment="center")
        st.info("💡 The more books you've rated, the better your recommendations!")

        st.markdown("---")

        st.markdown("## Click here to learn more")

        with st.expander("⭐ Recommendations", expanded=False):
            st.markdown("Personalized picks based on your Goodreads Reading List.")
            st.caption("- What should I read next?")
            st.caption("- Suggest books like my favorites.")
            st.caption("- Recommend something similar to Dune.")

        with st.expander("🔍 Check a book", expanded=False):
            st.markdown(
                "See if a specific title matches your taste. Just enter the title and author or ISBN"
            )
            st.caption("- Does Project Hail Mary fit my style?")
            st.caption("- Check 9780553386790 for me.")
            st.caption("- Does The Night Circus by Erin Morgenstern fit my taste?")

        with st.expander("📋 Search your to-read list", expanded=False):
            st.markdown("Find saved books by mood, theme, or vibe.")
            st.caption("- I am today in the mood for something romantic.")
            st.caption("- Show me sci-fi from my list.")
            st.caption("- Anything cozy and light from my reading list?")

        st.divider()

        if st.button(
            "↻ Restart BookBuddy", key="refresh_btn", use_container_width=True
        ):
            handle_refresh()
        st.caption("*Book Recommendation Assistant v1.0*")


def initialize_session_state() -> None:
    """Initialize session state defaults once per session."""
    if "chat_msgs" not in st.session_state:
        st.session_state.chat_msgs = []

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = False

    if "upload_in_progress" not in st.session_state:
        st.session_state.upload_in_progress = False

    if "upload_error" not in st.session_state:
        st.session_state.upload_error = None

    if "last_recommendations" not in st.session_state:
        st.session_state.last_recommendations = None

    if "last_checked_books" not in st.session_state:
        st.session_state.last_checked_books = None


def render_chat_history() -> None:
    """Render stored chat messages with avatars."""
    for message in st.session_state.chat_msgs:
        # Create custom chat message with icon
        icon = USER_ICON if message["role"] == "user" else ASSISTANT_ICON
        with st.chat_message(message["role"], avatar=icon):
            st.markdown(
                message["content"],
                text_alignment="right" if message["role"] == "user" else "left",
            )


def setup_page() -> None:
    """Set up the Streamlit page configuration and main layout."""
    st.set_page_config(
        page_title="Book Recommendation",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(TITLE_HTML, unsafe_allow_html=True)

    initialize_session_state()
    render_sidebar()
    render_chat_history()


def ensure_vector_store_ready() -> bool:
    """
    Ensure vector store is initialized or load from disk; render upload UI if needed.

    Checks session state for existing vector store. If not present, attempts to load
    from persist directory. If no stored vector store, provides UI for CSV upload,
    handles ingestion with progress updates, and sets session state on completion.

    Returns:
        bool: True if vector store is ready, False otherwise.
    """
    if st.session_state.vector_store_ready:
        return True

    if vector_store_exists(persist_directory=DEFAULT_PERSIST_DIRECTORY):
        st.session_state.vector_store = load_book_vector_store(
            persist_directory=DEFAULT_PERSIST_DIRECTORY
        )
        st.session_state.vector_store_ready = True
        return True

    st.subheader("Upload your Goodreads library to get started!")
    st.markdown(
        "Download your Goodreads export (CSV) and upload it here so I can tailor "
        "recommendations to your reading history. "
        "Only books marked with 'read' will be considered.\n\n"
        "Login into Goodreads and find your export [here](https://www.goodreads.com/review/import)."
    )
    st.caption(
        "Disclaimer: You need an active Goodreads account to download your export. "
        "Other CSV files with information about books cannot be used, the application is only "
        "optimized for Goodreads exports. The upload of the CSV file can take a few minutes "
        "depending on the amount of books, because the application needs to fetch additional "
        "information to give you better recommendations."
    )
    uploaded_file = st.file_uploader(
        "",
        type=["csv"],
    )

    if st.session_state.upload_error:
        st.error(st.session_state.upload_error)

    if uploaded_file and not st.session_state.upload_in_progress:
        progress_bar = st.progress(0)
        status_text = st.empty()
        stats_text = st.empty()
        start_time = time.time()
        st.session_state.upload_in_progress = True
        st.session_state.upload_error = None

        def on_progress(current: int, total: int, title: str) -> None:
            # Callback function to update the progress bar and status during ingestion
            progress = int((current / max(total, 1)) * 100)
            progress_bar.progress(progress)
            elapsed = max(time.time() - start_time, 0.01)
            rate = current / elapsed
            remaining = (total - current) / rate if rate else 0
            status_text.markdown(f"Processing {current} of {total}: **{title}**")
            stats_text.markdown(f"Estimated time remaining: {int(remaining)} seconds")

        try:
            vector_store, total = ingest_csv(
                uploaded_file,
                persist_directory=DEFAULT_PERSIST_DIRECTORY,
                progress_callback=on_progress,
            )
            st.session_state.vector_store = vector_store
            st.session_state.vector_store_ready = True
            st.session_state.upload_in_progress = False
            st.success(f"Uploaded {total} books. Library is ready.")
            st.rerun()
        except Exception:
            st.session_state.upload_in_progress = False
            st.session_state.upload_error = (
                "Hmm, something went wrong with that file. "
                "Make sure it's a Goodreads export CSV and try again."
            )
            st.rerun()

    return False
