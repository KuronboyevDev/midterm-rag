---
title: HiluxCare Support Assistant
emoji: 🛠️
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# 🛠️ HiluxCare Support — RAG Customer Support Assistant

A Retrieval-Augmented Generation (RAG) customer-support assistant that answers
questions from company documents **with citations**, and raises **support
tickets** in GitHub via **function calling**.

> Capstone project — Customer Support solution able to answer questions and raise
> support tickets.

## ✨ Features

- 💬 **Web chat** grounded in company documents (RAG).
- 📚 **Citations** — every grounded answer cites the document file name and page
  (e.g. *"Toyota Hilux Manual, p. 213"*).
- 🧰 **Function calling (required)** — the LLM uses two tools:
  - `search_knowledge_base(query)` — semantic search over the docs
  - `create_support_ticket(name, email, summary, description)` — files a GitHub Issue
- 🎫 **Support tickets** — when the answer isn't found (or the user asks), the
  assistant collects name/email/summary/description and opens a GitHub Issue.
- 🧠 **Conversation history** kept in the context window.
- 🏢 **Company-aware** — knows its name, contact info, hours and services.

## 🏗️ Architecture

```
User ⇄ Streamlit UI (app.py)
            │
        agent.py  ── OpenAI gpt-4o-mini, tool-calling loop, full history
            │
   ┌────────┴─────────┐
   ▼                  ▼
search_knowledge_base   create_support_ticket
   │                          │
rag/retriever.py         integrations/github_tickets.py
   │ (bge-small embeddings)   │ (GitHub Issues REST API)
   ▼
chroma_db/  ← built by rag/ingest.py from ./data
```

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11 (pinned in `requirements.txt`) |
| LLM + function calling | OpenAI `gpt-4o-mini` |
| Embeddings | `BAAI/bge-small-en-v1.5` (local, free) |
| Vector store | ChromaDB (persisted on disk) |
| PDF parsing | `pypdf` (page-level for citations) |
| Issue tracker | GitHub Issues |
| UI | Streamlit |
| Deployment | HuggingFace Spaces |

## 🚀 Local setup

```bash
# 1. Python 3.11 + dependencies
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt

# 2. Secrets
cp .env.example .env          # then edit .env with your keys

# 3. Add documents to ./data  (see data/README.md), then build the index
python -m rag.ingest

# 4. Run
streamlit run app.py
```

## 🔑 Environment variables

| Variable | Required | Purpose |
|----------|----------|---------|
| `OPENAI_API_KEY` | ✅ | Chat + function calling |
| `GITHUB_TOKEN` | for real tickets | PAT with `issues: write` / `repo` scope |
| `GITHUB_REPO` | for real tickets | `owner/repo` that receives tickets |
| `TICKETS_DRY_RUN` | optional | `true` simulates tickets (no GitHub call) |

If GitHub isn't configured (or `TICKETS_DRY_RUN=true`), ticket creation is
simulated so the app still demos end-to-end.

## ☁️ Deploy to HuggingFace Spaces

1. Create a **Streamlit** Space.
2. Push this repo to it (PDFs go through **Git LFS** — see `.gitattributes`).
   Commit the prebuilt `chroma_db/` so the Space doesn't re-embed on startup.
3. In **Settings → Variables and secrets**, add `OPENAI_API_KEY`, `GITHUB_TOKEN`,
   `GITHUB_REPO`.
4. The Space builds from `requirements.txt` / `runtime.txt` and launches `app.py`.

## 📁 Project layout

```
.
├── app.py                      # Streamlit UI
├── agent.py                    # OpenAI tool-calling loop + history
├── config.py                   # settings + company profile + system prompt
├── rag/
│   ├── ingest.py               # docs -> page chunks -> embeddings -> Chroma
│   ├── retriever.py            # semantic search + citations
│   ├── embeddings.py           # bge-small loader
│   └── tools.py                # OpenAI tool schemas + dispatch
├── integrations/
│   └── github_tickets.py       # GitHub Issues ticket creation
├── data/                       # source documents (+ included company FAQ)
├── chroma_db/                  # prebuilt vector store (commit this)
├── requirements.txt
├── runtime.txt
├── .env.example
├── .gitattributes              # Git LFS for PDFs
└── .gitignore
```

## ✅ Requirements coverage

- Web chat Q&A from datasources · ticket suggestion on no-answer · user-initiated
  tickets · ticket fields (name/email/summary/description) · GitHub issue tracker
  · citations (file + page) · conversation history · company awareness · ≥3 docs
  /≥2 PDF/≥1 PDF 400+ pages · Python + pinned deps · vector storage · function
  calling · Streamlit UI · HuggingFace Spaces deployment.
