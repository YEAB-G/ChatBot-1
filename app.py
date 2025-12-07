import os
from typing import List, Dict, Any

import streamlit as st
from groq import Groq
from pypdf import PdfReader

from rag_utils import RAGIndex


# ---------- CONFIG & STYLE ----------

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🔎",
    layout="wide",
)

# Global CSS for nicer UI
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: radial-gradient(circle at top left, #0f172a 0, #020617 45%, #020617 100%);
        color: #e5e7eb;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    /* Main content container */
    .main-block {
        background: rgba(15,23,42,0.85);
        border-radius: 18px;
        padding: 20px 24px;
        border: 1px solid rgba(148,163,184,0.25);
        box-shadow: 0 18px 60px rgba(15,23,42,0.8);
    }

    .header-title {
        font-size: 2rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.25rem;
    }

    .header-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 0.75rem;
    }

    .tag-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 4px 10px;
        border-radius: 999px;
        background: rgba(15,23,42,0.85);
        border: 1px solid rgba(148,163,184,0.5);
        font-size: 0.75rem;
        color: #e5e7eb;
        margin-right: 6px;
    }

    .sidebar .sidebar-content {
        background-color: #020617 !important;
    }

    /* Chat bubbles */
    .chat-bubble-user {
        background: linear-gradient(135deg, #2563eb, #4f46e5);
        color: white;
        padding: 10px 14px;
        border-radius: 16px;
        margin-bottom: 8px;
        max-width: 100%;
        font-size: 0.95rem;
    }

    .chat-bubble-assistant {
        background: rgba(15,23,42,0.95);
        border: 1px solid rgba(148,163,184,0.4);
        padding: 10px 14px;
        border-radius: 16px;
        margin-bottom: 8px;
        max-width: 100%;
        font-size: 0.95rem;
    }

    .chat-label {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9ca3af;
        margin-bottom: 3px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- LLM / RAG HELPERS ----------

def get_groq_client() -> Groq:
    """Create a Groq client using the GROQ_API_KEY environment variable."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is not set.\n\n"
            "Create a free API key in the Groq console, then set it:\n\n"
            "Windows PowerShell:\n"
            '  $env:GROQ_API_KEY="gsk-your-key-here"\n\n'
            "macOS / Linux:\n"
            '  export GROQ_API_KEY="gsk-your-key-here"'
        )
        st.stop()
    return Groq(api_key=api_key)


def build_system_message() -> str:
    return (
        "You are a helpful assistant that answers questions using ONLY the context "
        "provided. If the context is not enough, say you are not sure and suggest "
        "checking the original documents.\n\n"
        "Rules:\n"
        "- Do not invent facts that are not supported by the context.\n"
        "- If you are uncertain, say so.\n"
        "- Keep answers concise and well-structured."
    )


def build_user_message(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format context + question into a single user message."""
    if not retrieved_chunks:
        return (
            "There is no context available yet. "
            f"Still, the user asked: {question}\n\n"
            "Explain that there is no indexed data to answer from."
        )

    context_lines = []
    for r in retrieved_chunks:
        header = f"[{r['rank']}] From: {r['source_name']} (score={r['score']:.3f})"
        context_lines.append(header)
        context_lines.append(r["text"])
        context_lines.append("")

    context_block = "\n".join(context_lines)

    return (
        "Use the following context to answer the user's question.\n\n"
        "CONTEXT:\n"
        f"{context_block}\n\n"
        f"QUESTION: {question}\n\n"
        "Now provide a helpful answer based ONLY on this context."
    )


def call_llm(
    client: Groq,
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str = "llama-3.1-8b-instant",
    temperature: float = 0.2,
) -> str:
    """Call Groq Chat Completions to generate an answer."""
    system_msg = build_system_message()
    user_msg = build_user_message(question, retrieved_chunks)

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        temperature=temperature,
    )

    return response.choices[0].message.content.strip()


def init_session_state() -> None:
    """Initialize Streamlit session state variables."""
    if "index" not in st.session_state:
        st.session_state.index = None
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "last_results" not in st.session_state:
        st.session_state.last_results = []
    if "last_question" not in st.session_state:
        st.session_state.last_question = None


def extract_text_from_file(uploaded_file) -> str:
    """Handle txt, md, and pdf files and return plain text."""
    name = uploaded_file.name.lower()

    if name.endswith(".pdf"):
        pdf_reader = PdfReader(uploaded_file)
        pages_text = []
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            pages_text.append(page_text)
        return "\n".join(pages_text)

    # Default: treat as text
    raw_bytes = uploaded_file.read()
    try:
        return raw_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return raw_bytes.decode(errors="ignore")


# ---------- MAIN APP ----------

def main() -> None:
    init_session_state()

    # Top header block
    st.markdown(
        """
        <div class="main-block">
            <div class="header-title">
                🔎 RAG Chatbot
            </div>
            <div class="header-subtitle">
                Upload your own documents and chat with them. This app uses Retrieval-Augmented Generation:
                local embeddings + vector search + Groq Llama 3.1.
            </div>
            <div>
                <span class="tag-pill">RAG</span>
                <span class="tag-pill">PDF, TXT, MD</span>
                <span class="tag-pill">Groq · Llama 3.1</span>
                <span class="tag-pill">SentenceTransformers</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")  # small spacer

    # Sidebar: documents + index configuration
    with st.sidebar:
        st.header("📂 Documents & Index")

        uploaded_files = st.file_uploader(
            "Upload one or more documents",
            type=["txt", "md", "pdf"],
            accept_multiple_files=True,
        )

        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=200,
            max_value=1000,
            value=400,
            step=50,
        )
        chunk_overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=400,
            value=80,
            step=20,
        )

        embedding_model = st.text_input(
            "Embedding model (SentenceTransformers)",
            value="all-MiniLM-L6-v2",
        )

        if st.button("Build / Rebuild Vector Index", type="primary"):
            if not uploaded_files:
                st.warning("Please upload at least one file first.")
            else:
                index = RAGIndex(model_name=embedding_model)
                documents = []

                for f in uploaded_files:
                    text = extract_text_from_file(f)
                    documents.append({"name": f.name, "text": text})
                    index.add_document(
                        text=text,
                        source_name=f.name,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )

                st.session_state.index = index
                st.session_state.documents = documents
                st.session_state.last_results = []
                st.session_state.last_question = None

                st.success(
                    f"Index built ✅ — {len(documents)} document(s), {len(index.chunks)} chunks."
                )

        st.markdown("---")
        st.header("⚙️ Retrieval & LLM")

        top_k = st.slider(
            "Number of chunks to retrieve (top_k)",
            min_value=1,
            max_value=10,
            value=4,
        )
        temperature = st.slider(
            "LLM temperature (creativity)",
            min_value=0.0,
            max_value=1.0,
            value=0.2,
            step=0.05,
        )

        st.markdown("---")
        st.caption(
            "Tip: First build the index, then ask questions about your documents.\n"
            "Using Groq free tier (Llama 3.1) for generation."
        )

    # Layout: chat on the left, context on the right
    col_chat, col_context = st.columns([2, 1])

    # Left: chat UI
    with col_chat:
        st.markdown('<div class="main-block">', unsafe_allow_html=True)
        st.subheader("💬 Chat with your documents")

        for msg in st.session_state.chat_history:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                st.markdown(
                    '<div class="chat-label">You</div>'
                    f'<div class="chat-bubble-user">{content}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="chat-label">Assistant</div>'
                    f'<div class="chat-bubble-assistant">{content}</div>',
                    unsafe_allow_html=True,
                )

        user_question = st.chat_input("Ask something about your uploaded documents...")

        if user_question:
            if not st.session_state.index or st.session_state.index.is_empty():
                st.error("Please upload documents and build the index first.")
            else:
                # Record user message
                st.session_state.chat_history.append(
                    {"role": "user", "content": user_question}
                )

                st.markdown(
                    '<div class="chat-label">You</div>'
                    f'<div class="chat-bubble-user">{user_question}</div>',
                    unsafe_allow_html=True,
                )

                index = st.session_state.index
                results = index.search(user_question, top_k=top_k)
                st.session_state.last_results = results
                st.session_state.last_question = user_question

                client = get_groq_client()

                with st.spinner("Thinking with RAG..."):
                    try:
                        answer = call_llm(
                            client=client,
                            question=user_question,
                            retrieved_chunks=results,
                            temperature=temperature,
                        )
                    except Exception as e:
                        answer = (
                            "⚠️ Error while calling the Groq API:\n\n"
                            f"`{type(e).__name__}: {e}`"
                        )

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )
                st.markdown(
                    '<div class="chat-label">Assistant</div>'
                    f'<div class="chat-bubble-assistant">{answer}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("</div>", unsafe_allow_html=True)

    # Right: context/debug panel
    with col_context:
        st.markdown('<div class="main-block">', unsafe_allow_html=True)
        st.subheader("🔍 Retrieval & Context")

        if not st.session_state.last_results:
            st.info("Ask a question to see retrieved chunks and context here.")
        else:
            st.markdown("**Last question:**")
            st.write(st.session_state.last_question)

            st.markdown("**Top retrieved chunks (vector search):**")
            for r in st.session_state.last_results:
                with st.expander(
                    f"[{r['rank']}] {r['source_name']}  •  score={r['score']:.3f}"
                ):
                    st.write(r["text"])

        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
