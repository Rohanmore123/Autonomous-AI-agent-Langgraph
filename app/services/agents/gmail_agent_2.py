"""
app/services/agents/gmail_agent.py
===================================
Gmail Agent — read, search, send, and summarise emails using:
  • Google Gmail API v1 (via google-api-python-client)
  • Per-user OAuth2 tokens stored in PostgreSQL (GoogleOAuthToken)
  • LLMRouter for natural language understanding and email drafting

SECURITY:
  • Each user has their own OAuth2 credentials — strict isolation.
  • Tokens are refreshed automatically when expired.
  • We request only the minimum scopes needed:
      gmail.readonly  — read/search emails
      gmail.send      — send emails (only when user explicitly sends)
  • We NEVER store email body content in our DB — it stays in Google's systems.
    We only process it transiently in memory.

AGENT CAPABILITIES:
  1. search_emails(query)     — Gmail search syntax (from:, subject:, after:, etc.)
  2. get_email(message_id)    — fetch full email with body
  3. send_email(to, subject, body) — compose and send
  4. summarise_inbox(query)   — LLM-powered inbox summary
  5. draft_reply(message_id, instructions) — LLM drafts a reply
"""

from __future__ import annotations

import base64
import email as email_lib
from email.mime.text import MIMEText
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.logging import get_logger
from app.models.models import GoogleOAuthToken
from app.schemas.schemas import GmailEmailResponse
from app.services.llm.router import llm_router

logger = get_logger(__name__)


async def _get_user_credentials(user_id: str, db: AsyncSession) -> Credentials:
    """
    Load the user's Google OAuth2 credentials from DB and refresh if expired.
    Raises ValueError if user has not connected Google account.
    """
    result = await db.execute(
        select(GoogleOAuthToken).where(GoogleOAuthToken.user_id == user_id)
    )
    token_row = result.scalar_one_or_none()

    if not token_row:
        raise ValueError(
            "Google account not connected. Please authenticate via /api/v1/auth/google."
        )

    from app.core.config import get_settings
    cfg = get_settings()

    creds = Credentials(
        token=token_row.access_token,
        refresh_token=token_row.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cfg.google_oauth.client_id,
        client_secret=cfg.google_oauth.client_secret,
        scopes=token_row.scopes,
    )

    # Auto-refresh if expired
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Persist refreshed token
        token_row.access_token = creds.token
        from datetime import datetime, timezone
        token_row.token_expiry = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None
        await db.commit()
        logger.info("gmail.token.refreshed", user_id=user_id)

    return creds


def _build_gmail_service(creds: Credentials):
    """Build the Gmail API service object."""
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _parse_message(msg: dict[str, Any], include_body: bool = False) -> GmailEmailResponse:
    """Extract structured fields from a raw Gmail API message object."""
    headers = {h["name"]: h["value"] for h in msg.get("payload", {}).get("headers", [])}

    body: str | None = None
    if include_body:
        payload = msg.get("payload", {})
        parts = payload.get("parts", [])

        # Recursively search for text/plain part
        def extract_body(parts: list) -> str | None:
            for part in parts:
                if part.get("mimeType") == "text/plain":
                    data = part.get("body", {}).get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                if "parts" in part:
                    result = extract_body(part["parts"])
                    if result:
                        return result
            return None

        if parts:
            body = extract_body(parts)
        else:
            data = payload.get("body", {}).get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    to_header = headers.get("To", "")
    recipients = [r.strip() for r in to_header.split(",") if r.strip()]

    return GmailEmailResponse(
        message_id=msg["id"],
        thread_id=msg.get("threadId", ""),
        subject=headers.get("Subject", "(no subject)"),
        sender=headers.get("From", ""),
        recipients=recipients,
        date=headers.get("Date", ""),
        snippet=msg.get("snippet", ""),
        body=body,
    )


class GmailAgent:
    """
    Stateless Gmail agent.  Instantiated per request with user_id + db session.
    All Gmail API calls are synchronous (google-api-python-client is sync).
    In high-throughput systems, use run_in_executor to avoid blocking the event loop.
    """

    def __init__(self, user_id: str, db: AsyncSession) -> None:
        self.user_id = user_id
        self.db = db

    async def _service(self):
        creds = await _get_user_credentials(self.user_id, self.db)
        return _build_gmail_service(creds)

    async def search_emails(
        self, query: str, max_results: int = 10, include_body: bool = False
    ) -> list[GmailEmailResponse]:
        """
        Search emails using Gmail's powerful search syntax.
        Examples: 'from:boss@corp.com', 'subject:invoice after:2024/01/01'
        """
        import asyncio
        service = await self._service()

        def _sync_search():
            results = service.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()
            messages = results.get("messages", [])
            parsed = []
            for msg_ref in messages:
                full_msg = service.users().messages().get(
                    userId="me",
                    id=msg_ref["id"],
                    format="full" if include_body else "metadata",
                    metadataHeaders=["Subject", "From", "To", "Date"],
                ).execute()
                parsed.append(_parse_message(full_msg, include_body=include_body))
            return parsed

        loop = asyncio.get_event_loop()
        emails = await loop.run_in_executor(None, _sync_search)
        logger.info("gmail.search", user_id=self.user_id, query=query, found=len(emails))
        return emails

    async def get_email(self, message_id: str) -> GmailEmailResponse:
        """Fetch a single email by ID including full body."""
        import asyncio
        service = await self._service()

        def _sync_get():
            msg = service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            return _parse_message(msg, include_body=True)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync_get)

    async def send_email(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """
        Send an email. Returns the sent message ID.
        """
        import asyncio
        service = await self._service()

        msg = MIMEText(body, "plain", "utf-8")
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        if cc:
            msg["Cc"] = ", ".join(cc)
        if bcc:
            msg["Bcc"] = ", ".join(bcc)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        def _sync_send():
            sent = service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            return sent["id"]

        loop = asyncio.get_event_loop()
        message_id = await loop.run_in_executor(None, _sync_send)
        logger.info("gmail.sent", user_id=self.user_id, to=to, message_id=message_id)
        return message_id

    async def summarise_inbox(self, query: str = "in:inbox is:unread") -> str:
        """
        Fetch recent unread emails and ask the LLM to summarise them.
        Useful for a "morning briefing" agent task.
        """
        emails = await self.search_emails(query, max_results=15, include_body=False)

        if not emails:
            return "Your inbox is empty or no emails match the query."

        email_list = "\n\n".join(
            f"From: {e.sender}\nSubject: {e.subject}\nDate: {e.date}\nSnippet: {e.snippet}"
            for e in emails
        )

        llm_response = await llm_router.complete(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Please summarise these {len(emails)} emails in a concise digest. "
                        f"Group by priority (urgent / FYI / newsletters). "
                        f"Flag any that need a reply.\n\n{email_list}"
                    ),
                }
            ],
            system="You are an executive assistant who writes clear, structured email digests.",
        )

        return llm_response.content

    async def draft_reply(self, message_id: str, instructions: str) -> str:
        """
        Fetch an email and ask the LLM to draft a reply based on instructions.
        Returns draft text (not sent — user must review and call send_email).
        """
        original = await self.get_email(message_id)

        prompt = (
            f"Original email:\n"
            f"From: {original.sender}\n"
            f"Subject: {original.subject}\n"
            f"Body:\n{original.body or original.snippet}\n\n"
            f"Draft a reply following these instructions: {instructions}\n"
            f"Write ONLY the reply body — no greeting headers, no metadata."
        )

        llm_response = await llm_router.complete(
            messages=[{"role": "user", "content": prompt}],
            system="You are a professional email writer. Write clear, polite, concise emails.",
        )

        return llm_response.content