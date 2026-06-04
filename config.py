"""Central configuration for the Customer Support RAG assistant.

All tunables, paths, model names and company knowledge live here so the rest
of the codebase stays declarative and easy to audit.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Disable ChromaDB anonymous telemetry. Its bundled analytics lib has a buggy
# capture() signature, so every event prints a harmless "Failed to send
# telemetry event" line. Set the env var before chromadb is used, and silence
# the telemetry logger so nothing leaks even if it still tries.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# override=True so values in .env take precedence over any stale system
# environment variables (e.g. an old OPENAI_API_KEY set on the machine).
load_dotenv(override=True)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "support_kb"

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# --------------------------------------------------------------------------- #
# Retrieval / chunking
# --------------------------------------------------------------------------- #
CHUNK_TOKENS = 800          # target chunk size in tokens
CHUNK_OVERLAP_TOKENS = 120  # overlap between consecutive chunks
TOP_K = 5                   # chunks returned per search
# Minimum similarity for a chunk to be considered a real hit. Below this the
# assistant treats the knowledge base as "no answer found" and offers a ticket.
MIN_RELEVANCE = 0.30

# --------------------------------------------------------------------------- #
# Secrets / integration
# --------------------------------------------------------------------------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "")
TICKETS_DRY_RUN = os.getenv("TICKETS_DRY_RUN", "false").lower() in ("1", "true", "yes")


# --------------------------------------------------------------------------- #
# Company profile  (the "AI should know about the company" requirement)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Company:
    name: str = "HiluxCare Support"
    tagline: str = "Toyota Hilux sales, service & ownership support"
    email: str = "support@hiluxasia.com"
    phone: str = "+1 (800) 555-0142"
    hours: str = "Mon–Fri 08:00–18:00, Sat 09:00–14:00 (local time)"
    website: str = "https://hiluxasia.com"
    address: str = "120 Service Road, Springfield"
    products: tuple[str, ...] = field(
        default_factory=lambda: (
            "Toyota Hilux vehicle sales",
            "Scheduled maintenance & repairs",
            "Warranty support",
            "Parts & accessories",
        )
    )

    def as_prompt_block(self) -> str:
        """Render company facts for injection into the system prompt."""
        return (
            f"Company name: {self.name}\n"
            f"About: {self.tagline}\n"
            f"Support email: {self.email}\n"
            f"Support phone: {self.phone}\n"
            f"Business hours: {self.hours}\n"
            f"Website: {self.website}\n"
            f"Address: {self.address}\n"
            f"Services: {', '.join(self.products)}"
        )


COMPANY = Company()


def system_prompt() -> str:
    """Build the assistant's system prompt, grounded in the company profile."""
    return f"""You are the AI customer-support assistant for {COMPANY.name}.

ABOUT THE COMPANY (you may answer questions about this directly):
{COMPANY.as_prompt_block()}

HOW YOU WORK:
1. For any product/manual/policy question, you MUST call the
   `search_knowledge_base` tool before answering. Never invent facts about
   products, procedures, specifications or policies.
2. When you answer using retrieved content, ALWAYS cite your sources inline
   using the document file name and page, e.g. "(Toyota Hilux Manual, p. 213)".
   Cite every distinct source you used.
3. If the knowledge base returns nothing relevant, tell the user you could not
   find the answer and OFFER to create a support ticket so a human can help.
4. If the user asks to raise/open/create a ticket, OR agrees to your offer,
   collect: full name, email, a short summary (title) and a detailed
   description. Confirm the details, then call `create_support_ticket`.
   Do not fabricate the user's name or email — ask for them if missing.
5. Be concise, friendly and professional. You may answer company-profile
   questions (hours, contact, services) directly without searching.
"""
