# Main Streamlit app entry point with chat interface
from dotenv import load_dotenv

load_dotenv()

import streamlit as st

from ..utils.logging import setup_logging
from ..utils.streaming import stream_response
from .components import (
    ASSISTANT_ICON,
    USER_ICON,
    ensure_vector_store_ready,
    get_status_indicator,
    setup_page,
)

setup_logging()


def main():
    setup_page()
    if not ensure_vector_store_ready():
        return

    prompt = st.chat_input(
        "Try: 'Does Project Hail Mary fit my taste?' or 'What should I read next?'"
    )
    if prompt:
        st.session_state.chat_msgs.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_ICON):
            st.markdown(prompt, text_alignment="right")

        with st.chat_message("assistant", avatar=ASSISTANT_ICON):
            status_placeholder = st.empty()
            with status_placeholder.container():
                st.markdown(get_status_indicator("Thinking..."), unsafe_allow_html=True)

            response = st.write_stream(stream_response(prompt=prompt))

        assistant_content = "".join(response)
        message_data = {"role": "assistant", "content": assistant_content}
        st.session_state.chat_msgs.append(message_data)
        st.rerun()
