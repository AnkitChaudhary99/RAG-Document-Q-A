import hashlib
import os

import streamlit as st

from chatbot import Chatbot
from vectorstore import VectorStore


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide",
)


# ---------------------------------------------------------
# Custom styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>

    /* Main application width */
    .block-container {
        max-width: 1050px;
        padding-top: 3.5rem;
        padding-bottom: 7rem;
    }

    /* Hide Streamlit decoration */
    [data-testid="stDecoration"] {
        display: none;
    }

    /* Hero */
    .hero-eyebrow {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #8b8f98;
        margin-bottom: 0.6rem;
    }

    .hero-title {
        font-size: 3.3rem;
        line-height: 1.05;
        font-weight: 750;
        letter-spacing: -0.055em;
        margin: 0;
    }

    .hero-subtitle {
        color: #9da1aa;
        font-size: 1rem;
        margin-top: 1rem;
        margin-bottom: 2.5rem;
    }

    /* Section headings */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .section-description {
        color: #8e929b;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }

    /* Document card */
    .document-card {
        border: 1px solid #30343c;
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        background: #14171d;
        margin-bottom: 1rem;
    }

    .document-name {
        font-weight: 650;
        font-size: 1rem;
    }

    .document-meta {
        color: #858a94;
        font-size: 0.8rem;
        margin-top: 0.2rem;
    }

    /* Status cards */
    .status-card {
        border-radius: 12px;
        padding: 0.9rem 1rem;
        margin: 0.75rem 0 1.25rem 0;
        border: 1px solid #30343c;
        background: #14171d;
    }

    .status-success {
        border-color: #285b43;
        background: #10271d;
    }

    .status-info {
        border-color: #284e72;
        background: #102238;
    }

    .status-title {
        font-weight: 650;
    }

    .status-text {
        color: #9da1aa;
        font-size: 0.85rem;
        margin-top: 0.2rem;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.6rem;
    }

    /* Chat width */
    [data-testid="stChatMessage"] {
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Retrieved context */
    .source-label {
        font-weight: 650;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
    }

    .source-text {
        color: #a4a8b0;
        font-size: 0.85rem;
        line-height: 1.65;
    }

    /* Divider */
    hr {
        margin-top: 2.2rem;
        margin-bottom: 2.2rem;
        border-color: #242830;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #11141a;
    }

    .sidebar-brand {
        font-size: 1.15rem;
        font-weight: 700;
    }

    .sidebar-muted {
        color: #858a94;
        font-size: 0.82rem;
        line-height: 1.5;
    }

    .pipeline-step {
        color: #a1a5ad;
        font-size: 0.82rem;
        padding: 0.15rem 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

defaults = {
    "cohere_api_key": "",
    "pinecone_api_key": "",
    "api_keys_configured": False,
    "show_api_settings": True,
    "vectorstore": None,
    "chatbot": None,
    "chat_history": [],
    "document_hash": None,
    "processed_filename": None,
    "processed_chunks": 0,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_file_hash(uploaded_file) -> str:
    """Create a stable ID for the uploaded document."""
    file_bytes = uploaded_file.getvalue()
    return hashlib.md5(file_bytes).hexdigest()


def reset_document_state():
    """Clear the currently loaded document and conversation."""
    st.session_state.vectorstore = None
    st.session_state.chatbot = None
    st.session_state.chat_history = []
    st.session_state.document_hash = None
    st.session_state.processed_filename = None
    st.session_state.processed_chunks = 0


def connect_services(cohere_key: str, pinecone_key: str):
    """Save API credentials in session state."""
    st.session_state.cohere_api_key = cohere_key.strip()
    st.session_state.pinecone_api_key = pinecone_key.strip()
    st.session_state.api_keys_configured = True
    st.session_state.show_api_settings = False


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:

    st.markdown(
        '<div class="sidebar-brand">Document Q&A</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="sidebar-muted">Retrieval-augmented document intelligence</div>',
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------
    # API configuration
    # -----------------------------------------------------

    if st.session_state.api_keys_configured and not st.session_state.show_api_settings:

        st.markdown("### CONNECTION")

        st.success("Services connected.")

        if st.button(
            "Change API settings",
            use_container_width=True,
        ):
            st.session_state.show_api_settings = True
            st.rerun()

    else:

        st.markdown("### CONNECTION")

        # IMPORTANT:
        # These widgets deliberately do NOT use
        # key="cohere_api_key" or key="pinecone_api_key".
        #
        # This prevents Streamlit's
        # StreamlitWidgetAlreadyInstantiatedError.

        cohere_input = st.text_input(
            "Cohere API Key",
            value=st.session_state.cohere_api_key,
            type="password",
            placeholder="Enter Cohere API key",
        )

        pinecone_input = st.text_input(
            "Pinecone API Key",
            value=st.session_state.pinecone_api_key,
            type="password",
            placeholder="Enter Pinecone API key",
        )

        if st.button(
            "Connect services",
            type="primary",
            use_container_width=True,
        ):

            if not cohere_input.strip():
                st.error("Enter your Cohere API key.")

            elif not pinecone_input.strip():
                st.error("Enter your Pinecone API key.")

            else:
                connect_services(
                    cohere_input,
                    pinecone_input,
                )
                st.rerun()

    st.divider()

    # -----------------------------------------------------
    # Pipeline explanation
    # -----------------------------------------------------

    st.markdown("### PIPELINE")

    st.markdown(
        """
        <div class="pipeline-step">PDF extraction</div>
        <div class="pipeline-step">↓</div>
        <div class="pipeline-step">Text chunking</div>
        <div class="pipeline-step">↓</div>
        <div class="pipeline-step">Cohere embeddings</div>
        <div class="pipeline-step">↓</div>
        <div class="pipeline-step">Pinecone retrieval</div>
        <div class="pipeline-step">↓</div>
        <div class="pipeline-step">Cohere generation</div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # -----------------------------------------------------
    # Document status
    # -----------------------------------------------------

    st.markdown("### DOCUMENT")

    if st.session_state.processed_filename:

        st.markdown(
            f"""
            <div class="sidebar-muted">
                <strong>{st.session_state.processed_filename}</strong><br>
                {st.session_state.processed_chunks} indexed chunks
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(
            "Clear document",
            use_container_width=True,
        ):
            reset_document_state()
            st.rerun()

    else:

        st.markdown(
            '<div class="sidebar-muted">No document indexed yet.</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.chat_history:

        st.divider()

        if st.button(
            "Clear conversation",
            use_container_width=True,
        ):
            st.session_state.chat_history = []
            st.rerun()


# ---------------------------------------------------------
# Main hero
# ---------------------------------------------------------

st.markdown(
    '<div class="hero-eyebrow">DOCUMENT INTELLIGENCE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<h1 class="hero-title">Ask your documents.</h1>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
        Upload a PDF, index its contents, and ask questions with answers grounded in the source document.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# API configuration guard
# ---------------------------------------------------------

if not st.session_state.api_keys_configured:

    st.markdown(
        """
        <div class="status-card status-info">
            <div class="status-title">Connect your services</div>
            <div class="status-text">
                Enter your Cohere and Pinecone API keys in the sidebar to begin.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Source document section
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Source document</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="section-description">Upload a PDF to create a searchable document index.</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    label_visibility="collapsed",
)


# ---------------------------------------------------------
# Process document
# ---------------------------------------------------------

if uploaded_file is not None:

    current_hash = get_file_hash(uploaded_file)

    file_size_kb = len(uploaded_file.getvalue()) / 1024

    st.markdown(
        f"""
        <div class="document-card">
            <div class="document-name">📄 {uploaded_file.name}</div>
            <div class="document-meta">{file_size_kb:.1f} KB • PDF document</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if (
        st.session_state.document_hash
        and current_hash != st.session_state.document_hash
    ):
        reset_document_state()

    process_button = st.button(
        "Process document",
        type="primary",
        use_container_width=True,
        disabled=not st.session_state.api_keys_configured,
    )

    if process_button:

        if not st.session_state.api_keys_configured:
            st.error("Connect Cohere and Pinecone first.")

        else:

            try:

                with st.spinner(
                    "Extracting, chunking, embedding, and indexing document..."
                ):

                    file_path = "uploaded_document.pdf"

                    with open(file_path, "wb") as file:
                        file.write(uploaded_file.getvalue())

                    vectorstore = VectorStore(
                        pdf_path=file_path,
                        cohere_api_key=st.session_state.cohere_api_key,
                        pinecone_api_key=st.session_state.pinecone_api_key,
                        document_id=current_hash,
                    )

                    chatbot = Chatbot(
                        vectorstore=vectorstore,
                        cohere_api_key=st.session_state.cohere_api_key,
                    )

                    st.session_state.vectorstore = vectorstore
                    st.session_state.chatbot = chatbot
                    st.session_state.document_hash = current_hash
                    st.session_state.processed_filename = uploaded_file.name
                    st.session_state.processed_chunks = len(
                        vectorstore.chunks
                    )
                    st.session_state.chat_history = []

                st.success(
                    f"Document processed successfully — "
                    f"{len(vectorstore.chunks)} chunks indexed."
                )

            except Exception as error:

                st.error("Document processing failed.")

                with st.expander("Show technical details"):
                    st.exception(error)


# ---------------------------------------------------------
# Document status
# ---------------------------------------------------------

if st.session_state.vectorstore is not None:

    st.markdown(
        f"""
        <div class="status-card status-success">
            <div class="status-title">Document ready</div>
            <div class="status-text">
                {st.session_state.processed_filename}
                is indexed and ready for questions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.divider()


# ---------------------------------------------------------
# Conversation
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">Conversation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-description">
        Ask questions and inspect the source passages used to generate each answer.
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

if not st.session_state.chat_history:

    if st.session_state.vectorstore is None:

        st.info(
            "Upload a PDF and select “Process document” to get started."
        )

    else:

        st.info(
            "Your document is ready. Ask a question below."
        )

        st.markdown("#### Try asking")

        suggestion_columns = st.columns(3)

        suggestions = [
            "What is this document about?",
            "Summarize the main points.",
            "What are the key conclusions?",
        ]

        for column, suggestion in zip(
            suggestion_columns,
            suggestions,
        ):
            with column:
                st.caption(suggestion)


else:

    for item in st.session_state.chat_history:

        user_query = item["question"]
        answer = item["answer"]
        retrieved_docs = item["documents"]

        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):

            st.markdown(answer)

            if retrieved_docs:

                with st.expander(
                    f"Retrieved context · {len(retrieved_docs)} sections"
                ):

                    for index, document in enumerate(
                        retrieved_docs,
                        start=1,
                    ):

                        page_number = document.get("page")

                        if page_number:
                            source_label = (
                                f"Source {index} · Page {page_number}"
                            )
                        else:
                            source_label = f"Source {index}"

                        st.markdown(
                            f'<div class="source-label">{source_label}</div>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f"""
                            <div class="source-text">
                                {document["data"]["text"]}
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        if index < len(retrieved_docs):
                            st.divider()


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

if st.session_state.vectorstore is not None:

    user_question = st.chat_input(
        "Ask a question about your document..."
    )

    if user_question:

        try:

            with st.spinner("Searching the document..."):

                answer, retrieved_docs = (
                    st.session_state.chatbot.respond(
                        user_question
                    )
                )

            st.session_state.chat_history.append(
                {
                    "question": user_question,
                    "answer": answer,
                    "documents": retrieved_docs,
                }
            )

            st.rerun()

        except Exception as error:

            st.error("Unable to answer the question.")

            with st.expander("Show technical details"):
                st.exception(error)