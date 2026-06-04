"""Function-calling tool definitions and dispatch.

Exposes two tools to the LLM:
  * search_knowledge_base  -> semantic retrieval over the documents
  * create_support_ticket  -> files a GitHub Issue

`TOOL_SCHEMAS` is passed to the OpenAI Chat Completions API; `dispatch` executes
a tool call and returns (json_string_for_model, structured_payload_for_ui).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from integrations.github_tickets import create_ticket
from rag.retriever import search

# --------------------------------------------------------------------------- #
# Schemas advertised to the model
# --------------------------------------------------------------------------- #
TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "Search the company's document knowledge base (product manuals, "
                "warranty guides, support policies) for information needed to "
                "answer a user question. Returns relevant passages with their "
                "source file name and page number for citation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A focused search query describing the information needed.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_support_ticket",
            "description": (
                "Create a customer support ticket in the issue tracking system. "
                "Use only after collecting and confirming the user's real name, "
                "email, a short summary (title) and a detailed description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Full name of the user."},
                    "email": {"type": "string", "description": "Contact email of the user."},
                    "summary": {"type": "string", "description": "Short ticket title."},
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the issue or request.",
                    },
                },
                "required": ["name", "email", "summary", "description"],
            },
        },
    },
]


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #
def _run_search(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    query = args.get("query", "")
    hits = search(query)

    if not hits:
        payload = {"found": False, "results": []}
        model_text = json.dumps(
            {
                "found": False,
                "message": "No relevant information found in the knowledge base.",
            }
        )
        return model_text, payload

    results = [
        {
            "citation": h.citation,
            "source": h.source,
            "page": h.page,
            "score": round(h.score, 3),
            "content": h.text,
        }
        for h in hits
    ]
    model_text = json.dumps({"found": True, "results": results})
    payload = {"found": True, "results": results}
    return model_text, payload


def _run_ticket(args: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    result = create_ticket(
        name=args.get("name", ""),
        email=args.get("email", ""),
        summary=args.get("summary", ""),
        description=args.get("description", ""),
    )
    model_text = json.dumps(
        {
            "success": result.success,
            "message": result.message,
            "url": result.url,
            "number": result.number,
        }
    )
    payload = {
        "type": "ticket",
        "success": result.success,
        "message": result.message,
        "url": result.url,
        "number": result.number,
        "summary": args.get("summary", ""),
    }
    return model_text, payload


_HANDLERS = {
    "search_knowledge_base": _run_search,
    "create_support_ticket": _run_ticket,
}


def dispatch(name: str, arguments: str) -> Tuple[str, Dict[str, Any]]:
    """Execute a tool call.

    Returns (text_for_model, structured_payload_for_ui).
    """
    try:
        args = json.loads(arguments or "{}")
    except json.JSONDecodeError:
        args = {}

    handler = _HANDLERS.get(name)
    if handler is None:
        return json.dumps({"error": f"Unknown tool: {name}"}), {"error": name}
    return handler(args)
