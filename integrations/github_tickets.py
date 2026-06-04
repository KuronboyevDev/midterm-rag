"""Support-ticket integration: creates a GitHub Issue per ticket.

A ticket carries the user's name, email, a summary (issue title) and a detailed
description (issue body). If TICKETS_DRY_RUN is set, the ticket is simulated so
the app can be demoed without touching a real repository.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import requests

from config import GITHUB_REPO, GITHUB_TOKEN, TICKETS_DRY_RUN

GITHUB_API = "https://api.github.com"


@dataclass
class TicketResult:
    success: bool
    message: str
    url: Optional[str] = None
    number: Optional[int] = None


def _build_body(name: str, email: str, description: str) -> str:
    return (
        f"**Submitted by:** {name}\n"
        f"**Email:** {email}\n\n"
        f"---\n\n"
        f"{description}\n\n"
        f"---\n"
        f"_Created automatically by the HiluxCare AI support assistant._"
    )


def create_ticket(name: str, email: str, summary: str, description: str) -> TicketResult:
    """Create a support ticket as a GitHub Issue.

    Returns a structured result the assistant can relay to the user.
    """
    # --- validation ---
    missing = [
        label
        for label, value in (
            ("name", name),
            ("email", email),
            ("summary", summary),
            ("description", description),
        )
        if not (value and value.strip())
    ]
    if missing:
        return TicketResult(
            success=False,
            message=f"Cannot create ticket — missing required field(s): {', '.join(missing)}.",
        )

    body = _build_body(name, email, description)

    # --- dry run / safe demo mode ---
    if TICKETS_DRY_RUN or not (GITHUB_TOKEN and GITHUB_REPO):
        reason = "dry-run mode" if TICKETS_DRY_RUN else "GitHub not configured"
        return TicketResult(
            success=True,
            message=(
                f"Ticket simulated ({reason}). Title: '{summary}'. "
                "Set GITHUB_TOKEN, GITHUB_REPO and TICKETS_DRY_RUN=false to file real issues."
            ),
            url=None,
            number=None,
        )

    # --- real GitHub call ---
    url = f"{GITHUB_API}/repos/{GITHUB_REPO}/issues"
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {"title": summary, "body": body, "labels": ["support", "ai-created"]}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=20)
    except requests.RequestException as exc:
        return TicketResult(success=False, message=f"Network error contacting GitHub: {exc}")

    if resp.status_code == 201:
        data = resp.json()
        return TicketResult(
            success=True,
            message=f"Support ticket created successfully (#{data['number']}).",
            url=data.get("html_url"),
            number=data.get("number"),
        )

    detail = resp.json().get("message", resp.text) if resp.content else resp.reason
    return TicketResult(
        success=False,
        message=f"GitHub rejected the ticket (HTTP {resp.status_code}): {detail}",
    )
