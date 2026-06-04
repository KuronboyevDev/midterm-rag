"""The conversational agent: an OpenAI tool-calling loop.

Keeps the full conversation in `messages` (history stays in the context window),
lets the model call tools, feeds tool results back, and iterates until the model
produces a final answer. Surfaces the structured tool outputs (citations,
ticket results) so the UI can render them richly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

from openai import OpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL, system_prompt
from rag.tools import TOOL_SCHEMAS, dispatch

MAX_TOOL_ROUNDS = 5


@dataclass
class AgentTurn:
    """Result of one user turn."""
    answer: str
    sources: List[Dict[str, Any]] = field(default_factory=list)   # citation dicts
    tickets: List[Dict[str, Any]] = field(default_factory=list)   # ticket result dicts


def _client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env (local) or "
            "HuggingFace Space secrets (deployed)."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


def build_messages(history: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Prepend the system prompt to the running chat history."""
    return [{"role": "system", "content": system_prompt()}, *history]


def run_turn(history: List[Dict[str, str]]) -> AgentTurn:
    """Run one assistant turn over the given chat history.

    `history` is a list of {"role": "user"|"assistant", "content": str}.
    The final user message must already be appended.
    """
    client = _client()
    messages = build_messages(history)

    collected_sources: List[Dict[str, Any]] = []
    collected_tickets: List[Dict[str, Any]] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = response.choices[0].message

        # No tool calls -> final answer.
        if not msg.tool_calls:
            return AgentTurn(
                answer=msg.content or "",
                sources=_dedupe_sources(collected_sources),
                tickets=collected_tickets,
            )

        # Record the assistant's tool-call message, then execute each call.
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            model_text, payload = dispatch(tc.function.name, tc.function.arguments)

            if tc.function.name == "search_knowledge_base" and payload.get("found"):
                collected_sources.extend(payload["results"])
            elif tc.function.name == "create_support_ticket":
                collected_tickets.append(payload)

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": model_text,
                }
            )

    # Tool-round budget exhausted — ask the model for a final summary.
    final = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return AgentTurn(
        answer=final.choices[0].message.content or "",
        sources=_dedupe_sources(collected_sources),
        tickets=collected_tickets,
    )


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse duplicate (source, page) citations, keeping the best score."""
    best: Dict[str, Dict[str, Any]] = {}
    for s in sources:
        key = s["citation"]
        if key not in best or s["score"] > best[key]["score"]:
            best[key] = s
    return sorted(best.values(), key=lambda s: s["score"], reverse=True)
