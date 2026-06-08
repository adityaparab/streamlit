"""Dark-mode Streamlit chat interface for the RAG API.

Talks to the FastAPI `/extract` endpoint (with answer generation) to chat over
your ingested documents, and to `/ingest` to upload new PDFs.

Run with:
    uv run streamlit run streamlit_app.py
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("RAG_API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="centered")

# A little extra polish on top of the dark theme in .streamlit/config.toml.
st.markdown(
    """
    <style>
      .stApp { background-color: #0e1117; }
      .source-card {
          background: #1a1d29; border: 1px solid #2a2e3d;
          border-radius: 8px; padding: 10px 14px; margin: 6px 0;
          font-size: 0.85rem; color: #b9b9c6;
      }
      .source-card .meta { color: #7c5cff; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📚 RAG Chat")
st.caption(f"Connected to `{API_URL}` · LangChain + Qdrant + Ollama")

# --------------------------------------------------------------------------- #
# Sidebar: settings + document ingestion
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("⚙️ Settings")
    k = st.slider("Chunks to retrieve (k)", 1, 12, 4)
    show_sources = st.toggle("Show sources", value=True)

    st.divider()
    st.header("📥 Ingest a PDF")
    pdf = st.file_uploader("Upload PDF", type=["pdf"])
    clean = st.checkbox("Clean existing data first", value=False)
    if st.button("Ingest", use_container_width=True, disabled=pdf is None):
        with st.spinner("Ingesting…"):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    files={"file": (pdf.name, pdf.getvalue(), "application/pdf")},
                    data={"clean": str(clean).lower()},
                    timeout=600,
                )
                resp.raise_for_status()
                info = resp.json()
                st.success(
                    f"Ingested {info['chunks_ingested']} chunks from "
                    f"{info['source']}."
                )
            except requests.RequestException as exc:
                st.error(f"Ingestion failed: {exc}")

    st.divider()
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            for s in msg["sources"]:
                st.markdown(
                    f"<div class='source-card'><span class='meta'>"
                    f"{s['source']} · page {s['page']} · chunk {s['chunk_index']} "
                    f"· score {s['score']}</span><br>{s['content'][:300]}…</div>",
                    unsafe_allow_html=True,
                )

if prompt := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                resp = requests.post(
                    f"{API_URL}/extract",
                    json={"query": prompt, "k": k, "generate": True},
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data.get("answer") or "_No answer generated._"
                sources = data.get("matches", []) if show_sources else []
            except requests.RequestException as exc:
                answer, sources = f"⚠️ Request failed: {exc}", []

        st.markdown(answer)
        for s in sources:
            st.markdown(
                f"<div class='source-card'><span class='meta'>"
                f"{s['source']} · page {s['page']} · chunk {s['chunk_index']} "
                f"· score {s['score']}</span><br>{s['content'][:300]}…</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
