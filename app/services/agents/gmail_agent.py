"""
app/services/agents/gmail_agent.py  [REWRITTEN — LangGraph + @tool]
====================================================================
Autonomous Gmail Agent powered by LangGraph ReAct loop.

WHAT CHANGED:
  BEFORE: Direct method calls with fixed workflows
  AFTER:  LangGraph agent that autonomously plans email workflows

AUTONOMOUS CAPABILITIES:
  The agent can:
    1. Search and read emails based on natural language descriptions
    2. Understand context — read a thread before replying
    3. Draft replies and WAIT for user confirmation before sending
    4. Summarise inboxes and highlight action items
    5. Handle multi-step workflows:
       "Find the invoice email from last week and create a payment task"
       → search_emails → read_email → [switches to task agent work]
    6. Ask clarifying questions for ambiguous send requests

SAFETY DESIGN (Email Sending):
  send_email tool docstring explicitly instructs the LLM to:
  1. Draft first (draft_reply tool)
  2. Show user the draft
  3. Only call send_email AFTER explicit user confirmation
  
  This is enforced via the system prompt — the agent will ask
  "Shall I send this?" rather than sending immediately.

PRIVACY:
  Email bodies are NOT persisted to our database.
  They exist in memory only during the agent run.
"""
from __future__ import annotations
"""
app/services/agents/gmail_agent.py
====================================
Autonomous Gmail Agent — LangGraph ReAct loop.
Provider is driven by .env (PRIMARY_LLM_PROVIDER), no hardcoded Anthropic.
"""



import asyncio
import base64
from datetime import timezone

from langchain_core.messages import BaseMessage
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.models import GoogleOAuthToken
from app.schemas.schemas import GmailEmailResponse
from app.services.agents.graph import build_agent_graph, run_agent
from app.services.agents.tools.gmail_tools import build_gmail_tools

settings = get_settings()
logger = get_logger(__name__)


GMAIL_AGENT_SYSTEM_PROMPT = """You are an intelligent email assistant with direct access to the user's Gmail account.

You can autonomously:
- Search emails using Gmail's query syntax
- Read full email content and threads
- Summarise the inbox and highlight action items
- Draft replies based on user instructions (always show draft BEFORE sending)
- Send emails (ONLY after explicit user confirmation)

CRITICAL EMAIL SAFETY RULES:
1. NEVER send an email without showing the draft first and getting explicit confirmation.
2. When asked to "send", first call draft_reply, show the result, then ask:
   "Here's the draft — shall I send it?"
3. Only call send_email if the user explicitly says "yes, send it" or "go ahead".
4. For ambiguous recipient requests ("email my boss"), ask for clarification.

SENDING CONFIRMED EMAILS:
When the Manager passes an instruction like "User confirmed. Send the email to X with subject Y and body Z",
call send_email immediately with those exact details. Do not ask for confirmation again.

GMAIL QUERY SYNTAX EXAMPLES:
  "in:inbox is:unread"      → unread inbox
  "from:person@email.com"   → from specific sender
  "subject:keyword"         → subject contains word
  "after:2025/01/01"        → emails after a date
  "has:attachment"          → emails with attachments

Always be helpful, professional, and respectful of the user's email privacy."""


# ---------------------------------------------------------------------------
# LLM factory — reads provider from .env, no hardcoded Anthropic
# ---------------------------------------------------------------------------

def _build_llm(temperature: float = 0, max_tokens: int = 4096):
    provider = settings.llm.primary_provider
    model    = settings.llm.primary_model

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model,
            api_key=settings.llm.openai_api_key,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=settings.llm.google_ai_api_key,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
    # fallback: anthropic
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(
        model=model,
        api_key=settings.llm.anthropic_api_key,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ---------------------------------------------------------------------------
# Gmail credentials helper — shared by agent + tools
# ---------------------------------------------------------------------------

async def get_gmail_service(user_id: str, db: AsyncSession):
    """
    Load OAuth token from DB, refresh if expired, return a Gmail API service.
    Raises ValueError if no token found (user hasn't connected Google account).
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    result = await db.execute(
        select(GoogleOAuthToken).where(GoogleOAuthToken.user_id == user_id)
    )
    token_row = result.scalar_one_or_none()

    if not token_row:
        raise ValueError(
            "Google account not connected. "
            "Please authenticate via /api/v1/auth/google first."
        )

    creds = Credentials(
        token=token_row.access_token,
        refresh_token=token_row.refresh_token,
        token_uri="https://oauth2.googleapis.com/token",
        client_id=settings.google_oauth.client_id,
        client_secret=settings.google_oauth.client_secret,
        scopes=token_row.scopes,
    )

    # Refresh if expired
    if creds.expired and creds.refresh_token:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: creds.refresh(Request()))
        token_row.access_token = creds.token
        if creds.expiry:
            token_row.token_expiry = creds.expiry.replace(tzinfo=timezone.utc)
        await db.commit()
        logger.info("gmail.token.refreshed", user_id=user_id)

    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return service


# ---------------------------------------------------------------------------
# GmailAgent class
# ---------------------------------------------------------------------------

class GmailAgent:
    """
    LangGraph-powered autonomous Gmail agent.
    Provider-agnostic — reads PRIMARY_LLM_PROVIDER from .env.
    """

    def __init__(self, user_id: str, db: AsyncSession) -> None:
        self.user_id = user_id
        self.db      = db
        self.tools   = build_gmail_tools(user_id=user_id, db=db)

        llm = _build_llm(temperature=0, max_tokens=4096)
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.graph = build_agent_graph(self.llm_with_tools, self.tools)

    async def run(
        self,
        instruction: str,
        conversation_history: list[BaseMessage] | None = None,
    ) -> tuple[str, list[dict]]:
        """Run the autonomous Gmail agent. Returns (answer, tool_calls)."""
        return await run_agent(
            graph=self.graph,
            user_message=instruction,
            user_id=self.user_id,
            agent_type="gmail",
            system_prompt=GMAIL_AGENT_SYSTEM_PROMPT,
            conversation_history=conversation_history,
        )

    # ------------------------------------------------------------------
    # Direct helpers — used by gmail_tools internally (no agent loop)
    # ------------------------------------------------------------------

    async def search_emails(
        self,
        query: str,
        max_results: int = 10,
        include_body: bool = False,
    ) -> list[GmailEmailResponse]:
        """Fetch emails directly from Gmail API — no LLM involved."""
        service = await get_gmail_service(self.user_id, self.db)

        def _sync():
            results  = service.users().messages().list(
                userId="me", q=query, maxResults=max_results
            ).execute()
            messages = results.get("messages", [])
            parsed   = []
            for msg_ref in messages:
                full_msg = service.users().messages().get(
                    userId="me",
                    id=msg_ref["id"],
                    format="full" if include_body else "metadata",
                    metadataHeaders=["Subject", "From", "To", "Date"],
                ).execute()
                parsed.append(_parse_message(full_msg, include_body))
            return parsed

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    async def get_email(self, message_id: str) -> GmailEmailResponse:
        """Fetch a single email by ID with full body."""
        emails = await self.search_emails(
            f"rfc822msgid:{message_id}", max_results=1, include_body=True
        )
        if emails:
            return emails[0]

        # Fallback: fetch directly by Gmail message ID
        service = await get_gmail_service(self.user_id, self.db)

        def _sync():
            msg = service.users().messages().get(
                userId="me", id=message_id, format="full"
            ).execute()
            return _parse_message(msg, include_body=True)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _sync)

    async def send_email_direct(
        self,
        to: list[str],
        subject: str,
        body: str,
        cc: list[str] | None = None,
        bcc: list[str] | None = None,
    ) -> str:
        """Send an email. Returns sent message ID."""
        import email as email_lib
        from email.mime.text import MIMEText

        service = await get_gmail_service(self.user_id, self.db)

        msg = MIMEText(body)
        msg["to"]      = ", ".join(to)
        msg["subject"] = subject
        if cc:
            msg["cc"] = ", ".join(cc)
        if bcc:
            msg["bcc"] = ", ".join(bcc)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()

        def _sync():
            sent = service.users().messages().send(
                userId="me", body={"raw": raw}
            ).execute()
            return sent["id"]

        loop = asyncio.get_event_loop()
        msg_id = await loop.run_in_executor(None, _sync)
        logger.info("gmail.email.sent", user_id=self.user_id, to=to, subject=subject[:50])
        return msg_id


# ---------------------------------------------------------------------------
# Message parser
# ---------------------------------------------------------------------------

def _parse_message(msg: dict, include_body: bool = False) -> GmailEmailResponse:
    headers = {
        h["name"]: h["value"]
        for h in msg.get("payload", {}).get("headers", [])
    }

    body = None
    if include_body:
        def _extract(parts):
            for p in parts:
                if p.get("mimeType") == "text/plain":
                    data = p.get("body", {}).get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")
                if "parts" in p:
                    result = _extract(p["parts"])
                    if result:
                        return result
            return None

        payload = msg.get("payload", {})
        parts   = payload.get("parts", [])
        if parts:
            body = _extract(parts)
        else:
            data = payload.get("body", {}).get("data", "")
            if data:
                body = base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")

    to_header = headers.get("To", "")
    return GmailEmailResponse(
        message_id=msg["id"],
        thread_id=msg.get("threadId", ""),
        subject=headers.get("Subject", "(no subject)"),
        sender=headers.get("From", ""),
        recipients=[r.strip() for r in to_header.split(",") if r.strip()],
        date=headers.get("Date", ""),
        snippet=msg.get("snippet", ""),
        body=body,
    )