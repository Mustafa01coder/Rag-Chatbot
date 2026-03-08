# IMPORTS
import os
import json
import uuid
import hashlib
import tempfile
import time
import sqlite3
from datetime import datetime
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# CONFIG & LOAD ENV
load_dotenv()
BASE_DIR      = Path("rag_data")
DB_PATH       = BASE_DIR / "sessions.db"
CHROMA_DIR    = BASE_DIR / "chroma_index"
UPLOADS_DIR   = BASE_DIR / "uploads"

for d in [BASE_DIR, CHROMA_DIR, UPLOADS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="RAG Assistant Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# DATABASE SETUP (PERSISTENT SQLITE)
# ==========================================

def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id      TEXT NOT NULL,
            filename        TEXT NOT NULL,
            file_hash       TEXT NOT NULL,
            pages           INTEGER,
            chunks          INTEGER,
            uploaded_at     TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    con.commit()
    con.close()

def db_conn():
    return sqlite3.connect(DB_PATH)

def create_session(name: str) -> str:
    sid = str(uuid.uuid4())[:8]
    now = datetime.now().isoformat()
    con = db_conn()
    con.execute(
        "INSERT INTO sessions VALUES (?,?,?,?)",
        (sid, name, now, now)
    )
    con.commit()
    con.close()
    return sid


def get_all_sessions():
    con = db_conn()
    rows = con.execute(
        "SELECT id, name, created_at, updated_at FROM sessions ORDER BY updated_at DESC"
    ).fetchall()
    con.close()
    return rows

def delete_session(sid: str):
    con = db_conn()
    con.execute("DELETE FROM messages WHERE session_id=?", (sid,))
    con.execute("DELETE FROM uploaded_files WHERE session_id=?", (sid,))
    con.execute("DELETE FROM sessions WHERE id=?", (sid,))
    con.commit()
    con.close()

def rename_session(sid: str, new_name: str):
    con = db_conn()
    con.execute(
        "UPDATE sessions SET name=?, updated_at=? WHERE id=?",
        (new_name, datetime.now().isoformat(), sid)
    )
    con.commit()
    con.close()

def save_message(sid: str, role: str, content: str):
    now = datetime.now().isoformat()
    con = db_conn()
    con.execute(
        "INSERT INTO messages (session_id,role,content,timestamp) VALUES (?,?,?,?)",
        (sid, role, content, now)
    )
    con.execute(
        "UPDATE sessions SET updated_at=? WHERE id=?",
        (now, sid)
    )
    con.commit()
    con.close()

def load_messages(sid: str):
    con = db_conn()
    rows = con.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id=? ORDER BY id",
        (sid,)
    ).fetchall()
    con.close()
    return rows

def save_file_record(sid, filename, file_hash, pages, chunks):
    con = db_conn()
    con.execute(
        "INSERT INTO uploaded_files (session_id,filename,file_hash,pages,chunks,uploaded_at) VALUES (?,?,?,?,?,?)",
        (sid, filename, file_hash, pages, chunks, datetime.now().isoformat())
    )
    con.commit()
    con.close()

def get_session_files(sid: str):
    con = db_conn()
    rows = con.execute(
        "SELECT filename, file_hash, pages, chunks, uploaded_at FROM uploaded_files WHERE session_id=?",
        (sid,)
    ).fetchall()
    con.close()
    return rows

def file_already_indexed(file_hash: str) -> bool:
    con = db_conn()
    row = con.execute(
        "SELECT 1 FROM uploaded_files WHERE file_hash=?",
        (file_hash,)
    ).fetchone()
    con.close()
    return row is not None


# INIT DB

init_db()


# UI STYLING  (Dark editorial / refined)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg:        #09090b;
    --surface:   #111115;
    --border:    #27272a;
    --accent:    #6ee7b7;
    --accent2:   #818cf8;
    --text:      #e4e4e7;
    --muted:     #71717a;
    --danger:    #f87171;
}

* { box-sizing: border-box; }

html, body, .stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSelectbox div {
    background: #18181b !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

/* ---- Buttons ---- */
div.stButton > button {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text) !important;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    padding: 6px 16px;
    transition: all 0.2s;
    width: 100%;
}
div.stButton > button:hover {
    border-color: var(--accent);
    color: var(--accent) !important;
    box-shadow: 0 0 12px rgba(110,231,183,0.15);
}

/* Primary button override */
div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #6ee7b7, #818cf8);
    border: none;
    color: #09090b !important;
    font-weight: 700;
}

/* ---- Chat messages ---- */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border);
    border-radius: 12px;
    margin-bottom: 12px;
    padding: 14px 18px !important;
}

/* ---- Chat input ---- */
[data-testid="stChatInput"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: var(--text) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ---- Metrics ---- */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size:12px; }
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family:'JetBrains Mono',monospace; }

/* ---- File uploader ---- */
[data-testid="stFileUploader"] {
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: 12px;
    padding: 16px;
}

/* ---- Expander ---- */
details {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
}
details summary { color: var(--muted) !important; font-size: 13px; }

/* ---- Info/success banners ---- */
[data-testid="stAlert"] {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
}

/* ---- Headings ---- */
h1 { font-family:'Syne',sans-serif; font-weight:800; color: var(--text) !important; }
h2,h3 { font-family:'Syne',sans-serif; font-weight:700; color: var(--text) !important; }

/* ---- Session card ---- */
.session-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    margin-bottom: 8px;
    cursor: pointer;
    transition: border-color 0.2s;
}
.session-card:hover { border-color: var(--accent2); }
.session-card.active { border-color: var(--accent); }
.session-meta { font-size:11px; color: var(--muted); font-family:'JetBrains Mono',monospace; }

/* ---- Badge ---- */
.badge {
    display:inline-block;
    background: rgba(110,231,183,0.12);
    color: var(--accent);
    border-radius:6px;
    padding: 2px 8px;
    font-size:11px;
    font-family:'JetBrains Mono',monospace;
    margin-right:4px;
}
.badge-purple {
    background: rgba(129,140,248,0.12);
    color: var(--accent2);
}

/* ---- Divider ---- */
hr { border-color: var(--border) !important; }

/* ---- Code ---- */
code, .stCode { 
    background: #18181b !important; 
    color: var(--accent) !important;
    font-family:'JetBrains Mono',monospace !important;
    border-radius:6px;
}

/* ---- Slider ---- */
[data-testid="stSlider"] > div > div > div { background: var(--accent2) !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background: var(--muted); }

</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div style="padding: 32px 0 8px 0;">
  <h1 style="font-size:42px; margin:0; letter-spacing:-1px;">
    🧠 RAG Assistant <span style="color:#6ee7b7;">Pro</span>
  </h1>
  <p style="color:#71717a; font-size:15px; margin-top:6px; font-family:'JetBrains Mono',monospace;">
    Retrieval-Augmented Generation · Persistent Sessions · Semantic Search
  </p>
</div>
<hr>
""", unsafe_allow_html=True)


# SIDEBAR - SESSION MANAGER
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    api_key_input = st.text_input("Groq API Key", type="password", placeholder="")

    st.markdown("---")

    st.markdown("### 🗂 Sessions")

    # New session
    new_name = st.text_input("New session name", placeholder="e.g. Finance Q3 Docs")
    if st.button("＋ Create Session") and new_name.strip():
        sid = create_session(new_name.strip())
        st.session_state.active_session = sid
        st.rerun()

    st.markdown("---")

    # List sessions
    sessions = get_all_sessions()
    if sessions:
        st.markdown("**Existing Sessions**")
        for (sid, name, created, updated) in sessions:
            is_active = st.session_state.get("active_session") == sid
            card_class = "session-card active" if is_active else "session-card"
            updated_fmt = updated[:10]

            col_a, col_b = st.columns([4, 1])
            with col_a:
                if st.button(
                    f"{'▶ ' if is_active else ''}{name}",
                    key=f"sel_{sid}",
                    help=f"ID: {sid} | Updated: {updated_fmt}"
                ):
                    st.session_state.active_session = sid
                    st.rerun()
            with col_b:
                if st.button("🗑", key=f"del_{sid}", help="Delete session"):
                    delete_session(sid)
                    if st.session_state.get("active_session") == sid:
                        st.session_state.pop("active_session", None)
                    st.rerun()
    else:
        st.caption("No sessions yet. Create one above.")

    st.markdown("---")

    # RAG settings
    st.markdown("### 🔧 RAG Config")
    top_k = st.slider("Chunks Retrieved (k)", 1, 15, 5)
    chunk_size = st.slider("Chunk Size", 500, 2000, 1200, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 50, 400, 150, step=50)

    st.markdown("---")

# GUARD: API KEY + SESSION
api_key = api_key_input or os.getenv("GROQ_API_KEY")

if not api_key:
    st.warning("🔑 Enter your Groq API Key in the sidebar to get started.")
    st.stop()

if "active_session" not in st.session_state:
    st.info("👈 Create or select a session from the sidebar to begin.")
    st.stop()

active_sid = st.session_state.active_session

# Verify session still exists
sessions_list = get_all_sessions()
session_ids   = [s[0] for s in sessions_list]
if active_sid not in session_ids:
    st.session_state.pop("active_session", None)
    st.warning("Session not found. Please select another.")
    st.stop()

session_name = next(s[1] for s in sessions_list if s[0] == active_sid)

st.markdown(f"""
<div style="display:flex;align-items:center;gap:12px;margin-bottom:16px;">
  <span class="badge">Session</span>
  <span style="font-family:'JetBrains Mono',monospace;color:#e4e4e7;">{session_name}</span>
  <span class="badge badge-purple">{active_sid}</span>
</div>
""", unsafe_allow_html=True)

# CACHED EMBEDDINGS + LLM
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

@st.cache_resource(show_spinner=False)
def load_llm(_api_key: str):
    return ChatGroq(
        groq_api_key=_api_key,
        model_name="llama-3.3-70b-versatile"
    )

embeddings = load_embeddings()
llm        = load_llm(api_key)


# FILE UPLOAD + INDEXING
st.markdown("### 📂 Upload Documents")

uploaded_files = st.file_uploader(
    "Drop PDFs here — they'll be indexed and linked to this session",
    type="pdf",
    accept_multiple_files=True,
    label_visibility="visible"
)

# Show already-indexed files for this session
session_files = get_session_files(active_sid)
if session_files:
    st.markdown("**📋 Indexed in this session:**")
    cols = st.columns(3)
    for i, (fname, fhash, pages, chunks, uploaded_at) in enumerate(session_files):
        cols[i % 3].markdown(
            f"<div class='badge'>{fname}</div>"
            f"<span class='session-meta'> {pages}p · {chunks} chunks · {uploaded_at[:10]}</span>",
            unsafe_allow_html=True
        )

# Process new uploads
new_splits     = []
new_docs_count = 0

if uploaded_files:
    with st.spinner("Processing PDFs..."):
        for pdf in uploaded_files:

            file_bytes = pdf.getvalue()
            file_hash  = hashlib.md5(file_bytes).hexdigest()

            if file_already_indexed(file_hash):
                st.info(f"⚡ `{pdf.name}` already indexed — using cache.")
                continue

            # Write to temp file
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(file_bytes)
            tmp.close()

            loader = PyPDFLoader(tmp.name)
            docs   = loader.load()
            os.unlink(tmp.name)

            for d in docs:
                d.metadata["source_file"] = pdf.name
                d.metadata["session_id"]  = active_sid
                d.metadata["file_hash"]   = file_hash

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            splits = splitter.split_documents(docs)
            new_splits.extend(splits)
            new_docs_count += len(docs)

            save_file_record(active_sid, pdf.name, file_hash, len(docs), len(splits))

            # Save file to uploads dir
            save_path = UPLOADS_DIR / f"{file_hash}_{pdf.name}"
            save_path.write_bytes(file_bytes)

            st.success(f"✅ `{pdf.name}` — {len(docs)} pages, {len(splits)} chunks indexed.")


# VECTOR STORE
@st.cache_resource(show_spinner="Connecting to vector store...")
def get_vectorstore(_embeddings):
    if (CHROMA_DIR / "chroma.sqlite3").exists():
        return Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=_embeddings
        )
    return None

vectorstore = get_vectorstore(embeddings)

# Add new documents
if new_splits:
    if vectorstore is None:
        vectorstore = Chroma.from_documents(
            new_splits, embeddings,
            persist_directory=str(CHROMA_DIR)
        )
        # Clear cache so next load picks up new store
        get_vectorstore.clear()
    else:
        vectorstore.add_documents(new_splits)

    st.success(f"🗄️ Vector store updated — {len(new_splits)} new chunks added.")

# GUARD: NEED DOCUMENTS
all_session_files = get_session_files(active_sid)

if not all_session_files:
    st.info("Upload at least one PDF to start chatting.")
    st.stop()

if vectorstore is None:
    st.warning("Vector store not ready. Please re-upload your documents.")
    st.stop()

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": top_k, "fetch_k": top_k * 4}
)


# PROMPTS
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a query rewriter. Rewrite the user's question as a clear standalone search query based on the conversation history. Output ONLY the rewritten query, nothing else."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise document assistant. Answer ONLY using the provided context.\n"
     "Rules:\n"
     "- If the answer isn't in the context, say: 'Out of scope — not found in the uploaded documents.'\n"
     "- Cite the source file and page number when possible.\n"
     "- Be concise and structured. Use bullet points for lists.\n\n"
     "Context:\n{context}"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# LOAD PERSISTENT CHAT HISTORY
def build_langchain_history(sid: str) -> ChatMessageHistory:
    history = ChatMessageHistory()
    rows    = load_messages(sid)
    for role, content, _ in rows:
        if role == "human":
            history.add_user_message(content)
        else:
            history.add_ai_message(content)
    return history

# RENDER CHAT HISTORY
st.markdown("### 💬 Chat")

chat_rows = load_messages(active_sid)

for role, content, ts in chat_rows:
    avatar = "🧑" if role == "human" else "🤖"
    with st.chat_message("user" if role == "human" else "assistant"):
        st.markdown(content)
        st.caption(f"_{ts[:16].replace('T',' ')}_")

# CHAT INPUT
def join_docs(docs, max_chars=7000):
    text, total = "", 0
    for d in docs:
        if total + len(d.page_content) > max_chars:
            break
        text  += d.page_content + "\n\n---\n\n"
        total += len(d.page_content)
    return text


user_question = st.chat_input("Ask anything about your documents…")

# CHAT PIPELINE
if user_question:

    history = build_langchain_history(active_sid)

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_question)

    start = time.time()

    # Step 1 — rewrite query
    rewrite_msgs      = rewrite_prompt.format_messages(
        chat_history=history.messages,
        input=user_question
    )
    standalone_query  = llm.invoke(rewrite_msgs).content.strip()

    # Step 2 — retrieve docs
    docs    = retriever.invoke(standalone_query)
    context = join_docs(docs)

    # Step 3 — generate answer
    qa_msgs = qa_prompt.format_messages(
        chat_history=history.messages,
        input=user_question,
        context=context
    )

    with st.chat_message("assistant"):
        placeholder   = st.empty()
        full_response = ""
        for chunk in llm.stream(qa_msgs):
            full_response += chunk.content
            placeholder.markdown(full_response)
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.caption(f"_{ts}_")

    end = time.time()

    # Persist to DB
    save_message(active_sid, "human", user_question)
    save_message(active_sid, "ai",    full_response)

    # METRICS
    

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Chunks Indexed",  sum(f[3] for f in get_session_files(active_sid)))
    col2.metric("Retrieved Docs",  len(docs))
    col3.metric("Response Time",   f"{round(end - start, 2)}s")
    col4.metric("History Turns",   len(load_messages(active_sid)) // 2)

    # SOURCES
  
    sources = set()
    for d in docs:
        src = f"{d.metadata.get('source_file','?')} · Page {d.metadata.get('page','?')}"
        sources.add(src)

    if sources:
        st.markdown("##### 📚 Sources")
        scols = st.columns(min(len(sources), 3))
        for i, s in enumerate(sorted(sources)):
            scols[i % 3].info(s)


    # DEBUG
  

    with st.expander("🔍 Debug — Standalone Query"):
        st.code(standalone_query, language="text")

    with st.expander("📑 Retrieved Chunks"):
        for i, d in enumerate(docs, 1):
            st.markdown(
                f"**{i}. `{d.metadata.get('source_file','?')}` "
                f"(Page {d.metadata.get('page','?')})**"
            )
            st.write(d.page_content[:500])
            st.markdown("---")


# CLEAR CHAT BUTTON


if chat_rows:
    if st.button("🗑 Clear Chat History", key="clear_chat"):
        con = db_conn()
        con.execute("DELETE FROM messages WHERE session_id=?", (active_sid,))
        con.commit()
        con.close()
        st.rerun()


# FOOTER
st.markdown("""
<hr>
<div style="text-align:center;color:#52525b;font-family:'JetBrains Mono',monospace;font-size:12px;padding:16px 0 8px;">
  RAG Assistant Pro &nbsp;·&nbsp; Persistent SQLite + ChromaDB &nbsp;·&nbsp; Developed by Mustafa
</div>
""", unsafe_allow_html=True)