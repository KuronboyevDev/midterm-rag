"""HiluxCare Support — RAG customer-support assistant (Streamlit UI).

Features:
  * Web chat grounded in company documents (RAG) with inline citations.
  * Function calling: knowledge-base search + support-ticket creation.
  * Conversation history kept in the context window (session_state).
  * Company-aware system prompt (name, contact, hours, services).
"""
from __future__ import annotations

import base64
import html
from functools import lru_cache
from pathlib import Path

import streamlit as st

# Silence a benign warning where Streamlit's file-watcher tries to introspect
# torch.classes.__path__ (PyTorch custom-class registry). Cosmetic only.
try:
    import torch

    torch.classes.__path__ = []
except Exception:
    pass

from agent import run_turn
from config import CHROMA_DIR, COMPANY, OPENAI_API_KEY

st.set_page_config(
    page_title=f"{COMPANY.name} · AI Assistant",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

BG_IMAGE = Path(__file__).resolve().parent / "toyota-xilux.png"


@lru_cache(maxsize=1)
def _car_data_uri() -> str:
    """Base64-encode the Hilux image as a CSS/IMG data URI (cached)."""
    if not BG_IMAGE.exists():
        return ""
    encoded = base64.b64encode(BG_IMAGE.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"

# --------------------------------------------------------------------------- #
# Styling — production-grade design system
# --------------------------------------------------------------------------- #
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

      :root {
        --brand-900:#0a2540; --brand-700:#13386b; --brand-500:#2563eb;
        --brand-400:#3b82f6; --accent:#06b6d4;
        --ink:#0f172a; --muted:#64748b; --line:#e6ebf2;
        --ok:#059669; --ok-bg:#ecfdf5; --err:#dc2626; --err-bg:#fef2f2;
        --card:#ffffff; --surface:#f7f9fc;
      }

      html, body, [class*="css"], .stMarkdown, .stChatInput, button, input, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
      }

      .main .block-container { max-width: 880px; padding-top: 1.2rem; padding-bottom: 6rem; }
      #MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; }

      /* ---------------- Hero (dealership banner) ---------------- */
      .hero {
        position: relative; overflow: hidden;
        display:flex; align-items:center; gap:1.4rem;
        background:
          radial-gradient(900px 380px at 88% 18%, rgba(6,182,212,.30), transparent 60%),
          radial-gradient(700px 360px at 0% 120%, rgba(59,130,246,.30), transparent 55%),
          linear-gradient(125deg, var(--brand-900) 0%, var(--brand-700) 52%, #1d4ed8 120%);
        color:#fff; padding: 1.6rem 0 1.6rem 2rem; border-radius: 24px; margin:.2rem 0 1.4rem;
        box-shadow: 0 22px 48px -16px rgba(10,37,64,.55);
        border: 1px solid rgba(255,255,255,.08);
      }
      .hero-copy { flex:1 1 52%; min-width:0; z-index:2; }
      .hero-eyebrow {
        display:inline-flex; align-items:center; gap:.4rem; font-size:.72rem; font-weight:700;
        letter-spacing:.12em; text-transform:uppercase; color:#7dd3fc;
        background:rgba(125,211,252,.10); border:1px solid rgba(125,211,252,.25);
        padding:.28rem .65rem; border-radius:999px; margin-bottom:.7rem;
      }
      .hero h1 { margin:0; font-size:1.75rem; font-weight:800; letter-spacing:-.025em; line-height:1.12; }
      .hero h1 .accent { background:linear-gradient(90deg,#7dd3fc,#38bdf8);
                         -webkit-background-clip:text; background-clip:text; color:transparent; }
      .hero .sub { margin:.55rem 0 0; opacity:.92; font-size:.96rem; max-width:48ch; line-height:1.55; }
      .hero-chips { display:flex; flex-wrap:wrap; gap:.5rem; margin-top:1.1rem; }
      .chip {
        font-size:.76rem; font-weight:600; padding:.34rem .7rem; border-radius:999px;
        background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); color:#eaf2ff;
      }
      .hero-media {
        flex:1 1 48%; align-self:stretch; position:relative; min-height:210px;
        display:flex; align-items:center; justify-content:center;
      }
      /* radial "spotlight" so the white-cutout car blends onto the dark hero */
      .hero-media::before {
        content:""; position:absolute; inset:0;
        background:radial-gradient(closest-side at 60% 55%, rgba(255,255,255,.95), rgba(255,255,255,.55) 55%, transparent 78%);
        filter: blur(6px);
      }
      .hero-car {
        position:relative; z-index:2; width:108%; max-width:none; height:auto;
        transform:translateX(2%);
        filter: drop-shadow(0 26px 26px rgba(0,0,0,.40));
      }
      @media (max-width: 760px) {
        .hero { flex-direction:column; align-items:flex-start; padding:1.4rem 1.4rem 0; }
        .hero-car { width:100%; transform:none; }
      }

      /* ---------------- Sidebar ---------------- */
      section[data-testid="stSidebar"] { background:#0a2540; border-right:none; }
      section[data-testid="stSidebar"] * { color:#dbe7f5 !important; }
      .sb-card {
        background:linear-gradient(160deg, rgba(37,99,235,.22), rgba(6,182,212,.10));
        border:1px solid rgba(255,255,255,.10); border-radius:18px; padding:1.1rem;
        text-align:center; margin-bottom:1rem;
      }
      .sb-avatar {
        width:60px; height:60px; margin:0 auto .6rem; border-radius:50%;
        display:grid; place-items:center; font-size:1.8rem;
        background:linear-gradient(135deg,#2563eb,#06b6d4);
        box-shadow:0 8px 18px -6px rgba(6,182,212,.6);
      }
      .sb-name { font-weight:800; font-size:1.05rem; }
      .sb-tag { font-size:.78rem; opacity:.75; margin-top:.15rem; }
      .sb-row {
        display:flex; align-items:center; gap:.6rem; font-size:.84rem;
        padding:.5rem .2rem; border-bottom:1px solid rgba(255,255,255,.06);
      }
      .sb-row .ic {
        width:30px; height:30px; flex:0 0 auto; border-radius:9px; display:grid; place-items:center;
        background:rgba(255,255,255,.08); font-size:.95rem;
      }
      .sb-row .lbl { opacity:.6; font-size:.7rem; text-transform:uppercase; letter-spacing:.04em; }
      .sb-row .val { font-weight:600; }
      .sb-title { font-size:.72rem; text-transform:uppercase; letter-spacing:.08em;
                  opacity:.6; margin:1.2rem 0 .5rem; }
      .sb-cap { display:flex; gap:.55rem; align-items:flex-start; font-size:.84rem; margin:.45rem 0; }
      .sb-cap .dot { color:#06b6d4 !important; font-weight:800; }

      section[data-testid="stSidebar"] .stButton button {
        background:rgba(255,255,255,.08); color:#fff !important; border:1px solid rgba(255,255,255,.18);
        border-radius:12px; font-weight:600; transition:.15s;
      }
      section[data-testid="stSidebar"] .stButton button:hover {
        background:rgba(220,38,38,.85); border-color:transparent;
      }

      /* ---------------- Chat ---------------- */
      [data-testid="stChatMessage"] {
        background:var(--card); border:1px solid var(--line); border-radius:16px;
        padding:1rem 1.15rem; margin-bottom:.85rem;
        box-shadow:0 1px 2px rgba(15,23,42,.04);
      }
      [data-testid="stChatMessageContent"] { font-size:.96rem; line-height:1.6; color:var(--ink); }

      /* ---------------- Sources ---------------- */
      [data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:14px !important; }
      .src-item { padding:.55rem 0; border-bottom:1px dashed var(--line); }
      .src-item:last-child { border-bottom:none; }
      .src-head { display:flex; align-items:center; gap:.5rem; margin-bottom:.3rem; }
      .src-pill {
        display:inline-flex; align-items:center; gap:.35rem; background:#eff5ff; color:#1d4ed8;
        border:1px solid #dbe6fe; border-radius:8px; padding:.18rem .55rem;
        font-size:.78rem; font-weight:700;
      }
      .src-score { font-size:.72rem; color:var(--muted); font-weight:600; }
      .bar { height:5px; background:#eef2f7; border-radius:999px; overflow:hidden; width:120px; }
      .bar > span { display:block; height:100%; background:linear-gradient(90deg,#2563eb,#06b6d4); }
      .src-snippet { font-size:.82rem; color:var(--muted); margin-top:.35rem; line-height:1.5; }

      /* ---------------- Tickets ---------------- */
      .ticket-card {
        position:relative; border-radius:14px; padding:.95rem 1.1rem; margin:.55rem 0;
        background:linear-gradient(180deg,#f0f7ff,#ffffff); border:1px solid #d8e6fb;
        box-shadow:0 6px 18px -10px rgba(37,99,235,.5);
      }
      .ticket-card::before {
        content:""; position:absolute; left:0; top:0; bottom:0; width:5px;
        background:linear-gradient(180deg,#2563eb,#06b6d4); border-radius:14px 0 0 14px;
      }
      .ticket-card.err { background:var(--err-bg); border-color:#f6cccc; box-shadow:none; }
      .ticket-card.err::before { background:var(--err); }
      .ticket-head { font-weight:800; font-size:.95rem; color:var(--ink); display:flex; gap:.45rem; align-items:center; }
      .ticket-sum { font-weight:600; margin:.25rem 0 .1rem; }
      .ticket-msg { font-size:.84rem; color:var(--muted); }
      .ticket-link { font-size:.82rem; font-weight:700; color:#1d4ed8; text-decoration:none; }

      /* ---------------- Empty state suggestions ---------------- */
      .empty-title { font-size:.78rem; font-weight:700; text-transform:uppercase;
                     letter-spacing:.08em; color:var(--muted); margin:.4rem 0 .6rem; }
      .stButton button {
        border-radius:12px; border:1px solid var(--line); background:#fff; color:var(--ink);
        font-weight:600; font-size:.86rem; padding:.7rem .9rem; transition:.15s; text-align:left;
      }
      .stButton button:hover {
        border-color:var(--brand-400); color:var(--brand-500);
        box-shadow:0 6px 16px -8px rgba(37,99,235,.5); transform:translateY(-1px);
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Sidebar — company profile
# --------------------------------------------------------------------------- #
def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            f"""
            <div class="sb-card">
              <div class="sb-avatar">🛠️</div>
              <div class="sb-name">{html.escape(COMPANY.name)}</div>
              <div class="sb-tag">{html.escape(COMPANY.tagline)}</div>
            </div>
            <div class="sb-row"><div class="ic">📧</div>
              <div><div class="lbl">Email</div><div class="val">{html.escape(COMPANY.email)}</div></div></div>
            <div class="sb-row"><div class="ic">📞</div>
              <div><div class="lbl">Phone</div><div class="val">{html.escape(COMPANY.phone)}</div></div></div>
            <div class="sb-row"><div class="ic">🕒</div>
              <div><div class="lbl">Hours</div><div class="val">{html.escape(COMPANY.hours)}</div></div></div>
            <div class="sb-row"><div class="ic">🌐</div>
              <div><div class="lbl">Website</div><div class="val">{html.escape(COMPANY.website)}</div></div></div>

            <div class="sb-title">What I can do</div>
            <div class="sb-cap"><span class="dot">›</span><span>Answer questions from the product manuals &amp; policies</span></div>
            <div class="sb-cap"><span class="dot">›</span><span>Cite the document and page for every answer</span></div>
            <div class="sb-cap"><span class="dot">›</span><span>Open a support ticket when you need a human</span></div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        if st.button("🗑️  Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.session_state.render = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"📚  Sources & citations ({len(sources)})", expanded=False):
        for s in sources:
            pct = int(max(0.0, min(1.0, s["score"])) * 100)
            snippet = html.escape(s["content"].strip().replace("\n", " "))
            if len(snippet) > 300:
                snippet = snippet[:300] + "…"
            st.markdown(
                f"""
                <div class="src-item">
                  <div class="src-head">
                    <span class="src-pill">📄 {html.escape(s['citation'])}</span>
                    <span class="bar"><span style="width:{pct}%"></span></span>
                    <span class="src-score">{pct}% match</span>
                  </div>
                  <div class="src-snippet">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_tickets(tickets: list[dict]) -> None:
    for t in tickets:
        if t.get("success"):
            num = f"#{t['number']} " if t.get("number") else ""
            link = (
                f"<br><a class='ticket-link' href='{t['url']}' target='_blank'>View issue ↗</a>"
                if t.get("url") else ""
            )
            st.markdown(
                f"""
                <div class="ticket-card">
                  <div class="ticket-head">🎫 Support ticket {num}created</div>
                  <div class="ticket-sum">{html.escape(t.get('summary',''))}</div>
                  <div class="ticket-msg">{html.escape(t['message'])}{link}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="ticket-card err">
                  <div class="ticket-head">⚠️ Ticket not created</div>
                  <div class="ticket-msg">{html.escape(t['message'])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "history" not in st.session_state:
    st.session_state.history = []   # raw {role, content} for the LLM context window
if "render" not in st.session_state:
    st.session_state.render = []    # enriched turns for the UI (with sources/tickets)


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #
render_sidebar()

_car = _car_data_uri()
_car_html = f'<img class="hero-car" src="{_car}" alt="Toyota Hilux"/>' if _car else ""
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-copy">
        <span class="hero-eyebrow">🛠️ {html.escape(COMPANY.name)}</span>
        <h1>Your <span class="accent">Toyota Hilux</span><br>support, answered instantly.</h1>
        <p class="sub">Ask about your vehicle, warranty or service. I am official AI admin I can help you with your all questions and can raise a support ticket for you.</p>
        <div class="hero-chips">
          <span class="chip">📚 Grounded in your documents</span>
          <span class="chip">🔖 Cited sources &amp; pages</span>
          <span class="chip">🎫 Instant support tickets</span>
        </div>
      </div>
      <div class="hero-media">{_car_html}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not OPENAI_API_KEY:
    st.warning("⚠️ `OPENAI_API_KEY` is not configured. Add it to `.env` or Space secrets to start chatting.")
if not CHROMA_DIR.exists():
    st.error("❌ Knowledge base not found. Run `python -m rag.ingest` to build the vector store.")

# Replay history
for turn in st.session_state.render:
    avatar = "🧑" if turn["role"] == "user" else "🛠️"
    with st.chat_message(turn["role"], avatar=avatar):
        st.markdown(turn["content"])
        if turn["role"] == "assistant":
            render_sources(turn.get("sources", []))
            render_tickets(turn.get("tickets", []))

# Empty-state suggestions
if not st.session_state.render:
    st.markdown("<div class='empty-title'>✨ Try asking</div>", unsafe_allow_html=True)
    cols = st.columns(3)
    examples = [
        "How do I check the engine oil level?",
        "What does the Hilux warranty cover?",
        "I want to open a support ticket.",
    ]
    for col, ex in zip(cols, examples):
        if col.button(ex, use_container_width=True):
            st.session_state._pending = ex
            st.rerun()


# --------------------------------------------------------------------------- #
# Handle input
# --------------------------------------------------------------------------- #
prompt = st.chat_input("Type your question…")
if "_pending" in st.session_state:
    prompt = st.session_state.pop("_pending")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    st.session_state.render.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🛠️"):
        with st.spinner("Thinking…"):
            try:
                turn = run_turn(st.session_state.history)
                answer = turn.answer or "_(no response)_"
            except Exception as exc:  # surface errors cleanly in the UI
                answer = f"⚠️ Something went wrong: `{exc}`"
                turn = None

        st.markdown(answer)
        if turn:
            render_sources(turn.sources)
            render_tickets(turn.tickets)

    st.session_state.history.append({"role": "assistant", "content": answer})
    st.session_state.render.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": turn.sources if turn else [],
            "tickets": turn.tickets if turn else [],
        }
    )
