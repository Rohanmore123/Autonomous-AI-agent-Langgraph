"""
app/services/agents/tools/gmail_tools.py
==========================================
LangChain @tool decorated Gmail tools.

TOOLS:
  search_emails      — search Gmail with query syntax
  read_email         — fetch full email body by message ID
  send_email         — compose and send an email
  summarise_inbox    — AI-powered inbox digest
  draft_reply        — AI-drafted reply text for review
  mark_as_read       — mark email(s) as read
  get_email_thread   — get all messages in a thread

PRIVACY DESIGN:
  Email body content is NEVER persisted to our database.
  It is fetched from Google's servers, processed in-memory by the LLM,
  and the result (summary/draft) returned to the user.
  Only metadata (message IDs, subject, sender) may appear in audit logs.

RATE LIMITING:
  Gmail API has quotas (250 quota units/user/second).
  We don't exceed this in normal usage, but implement exponential back-off
  via tenacity for any quota errors.
"""


from __future__ import annotations


"""
app/services/agents/tools/gmail_tools.py
==========================================
LangChain @tool decorated Gmail tools.

FIXES vs previous version:
  1. Imports from gmail_agent (not the non-existent gmail_agent_2)
  2. summarise_inbox no longer calls agent.summarise_inbox() — infinite recursion fixed.
     It fetches emails directly and returns them as structured data for the LLM to summarise.
  3. send_email calls agent.send_email_direct() — the safe direct send method.
  4. draft_reply generates the draft text itself — no recursive agent.run() call.
"""


import asyncio
import json
from typing import Optional

from langchain_core.tools import tool, ToolException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger

logger = get_logger(__name__)


def build_gmail_tools(user_id: str, db: AsyncSession) -> list:
    """
    Factory: create Gmail tools bound to user_id and DB session.
    All tools use GmailAgent.search_emails() / get_email() directly —
    no recursive agent.run() calls.
    """

    async def _agent():
        """Get a GmailAgent instance for direct API calls (no LLM loop)."""
        from app.services.agents.gmail_agent import GmailAgent   # correct import
        return GmailAgent(user_id=user_id, db=db)

    # ------------------------------------------------------------------
    # Tool 1: search_emails
    # ------------------------------------------------------------------

    @tool
    async def search_emails(
        query: str,
        max_results: int = 10,
        include_body: bool = False,
    ) -> str:
        """
        Search Gmail using Gmail's query syntax.

        Examples:
          "in:inbox is:unread"              → unread inbox
          "from:boss@company.com"           → from a specific sender
          "subject:invoice after:2025/01/01"→ invoices since Jan 2025
          "has:attachment filename:pdf"     → emails with PDF attachments

        Args:
            query:        Gmail search query string.
            max_results:  Max emails to return (1-50, default 10).
            include_body: Fetch full body (slower). Use False for listing.

        Returns:
            JSON string with matching emails and metadata.
        """
        max_results = max(1, min(50, max_results))

        try:
            agent  = await _agent()
            emails = await agent.search_emails(query, max_results, include_body)
        except ValueError as e:
            raise ToolException(str(e))
        except Exception as e:
            logger.error("tool.search_emails.error", error=str(e), user_id=user_id)
            raise ToolException(f"Gmail search failed: {e}")

        if not emails:
            return json.dumps({
                "found": 0, "query": query, "emails": [],
                "message": "No emails found matching this query.",
            })

        return json.dumps({
            "found":  len(emails),
            "query":  query,
            "emails": [
                {
                    "message_id": e.message_id,
                    "thread_id":  e.thread_id,
                    "subject":    e.subject,
                    "from":       e.sender,
                    "to":         e.recipients,
                    "date":       e.date,
                    "snippet":    e.snippet,
                    "body":       e.body if include_body else None,
                }
                for e in emails
            ],
        })

    # ------------------------------------------------------------------
    # Tool 2: read_email
    # ------------------------------------------------------------------

    @tool
    async def read_email(message_id: str) -> str:
        """
        Fetch the full content of a specific email by its Gmail message ID.
        Get message_id from search_emails results.

        Args:
            message_id: Gmail message ID string.

        Returns:
            JSON string with full email including body text.
        """
        try:
            agent = await _agent()
            email = await agent.get_email(message_id)
        except ValueError as e:
            raise ToolException(str(e))
        except Exception as e:
            logger.error("tool.read_email.error", error=str(e), user_id=user_id)
            raise ToolException(f"Failed to read email: {e}")

        return json.dumps({
            "message_id": email.message_id,
            "thread_id":  email.thread_id,
            "subject":    email.subject,
            "from":       email.sender,
            "to":         email.recipients,
            "date":       email.date,
            "snippet":    email.snippet,
            "body":       email.body or "(No text body found)",
        })

    # ------------------------------------------------------------------
    # Tool 3: send_email
    # ------------------------------------------------------------------

    @tool
    async def send_email(
        to: list[str],
        subject: str,
        body: str,
        cc: Optional[list[str]] = None,
        bcc: Optional[list[str]] = None,
    ) -> str:
        """
        Send an email on behalf of the user.

        IMPORTANT: Always show a draft and get explicit user confirmation first.
        Only call this after the user says "yes, send it" or "go ahead".

        Args:
            to:      List of recipient email addresses (at least one required).
            subject: Email subject line.
            body:    Email body (plain text).
            cc:      Optional CC recipients.
            bcc:     Optional BCC recipients.

        Returns:
            JSON confirming delivery with sent message ID.
        """
        if not to:
            raise ToolException("At least one recipient (to) is required.")
        if not subject.strip():
            raise ToolException("Email subject cannot be empty.")
        if not body.strip():
            raise ToolException("Email body cannot be empty.")

        try:
            agent      = await _agent()
            message_id = await agent.send_email_direct(
                to=to, subject=subject, body=body,
                cc=cc or [], bcc=bcc or [],
            )
        except ValueError as e:
            raise ToolException(str(e))
        except Exception as e:
            logger.error("tool.send_email.error", error=str(e), user_id=user_id)
            raise ToolException(f"Failed to send email: {e}")

        return json.dumps({
            "success": True, "message_id": message_id,
            "to": to, "subject": subject, "status": "sent",
        })

    # ------------------------------------------------------------------
    # Tool 4: summarise_inbox
    # FIX: returns raw email data — LLM in the ReAct loop does the summarising.
    # Old version called agent.summarise_inbox() → agent.run() = infinite recursion.
    # ------------------------------------------------------------------

    @tool
    async def summarise_inbox(
        query: str = "in:inbox is:unread newer_than:3d",
        max_emails: int = 15,
    ) -> str:
        """
        Fetch recent inbox emails and return them for summarisation.
        The LLM will read this data and produce a natural language summary.

        Args:
            query:      Gmail query (default: recent unread inbox).
            max_emails: Max emails to fetch (default 15).

        Returns:
            JSON with email list. Summarise this data in your response.
        """
        max_emails = max(1, min(30, max_emails))

        try:
            agent  = await _agent()
            emails = await agent.search_emails(query, max_emails, include_body=False)
        except ValueError as e:
            raise ToolException(str(e))
        except Exception as e:
            logger.error("tool.summarise_inbox.error", error=str(e), user_id=user_id)
            raise ToolException(f"Failed to fetch inbox: {e}")

        if not emails:
            return json.dumps({
                "email_count": 0, "query": query,
                "message": "Inbox is empty or no emails match the query.",
                "emails": [],
            })

        return json.dumps({
            "email_count": len(emails),
            "query":       query,
            "instruction": "Summarise these emails by priority. Flag action items.",
            "emails": [
                {
                    "from":    e.sender,
                    "subject": e.subject,
                    "date":    e.date,
                    "snippet": e.snippet,
                }
                for e in emails
            ],
        })

    # ------------------------------------------------------------------
    # Tool 5: draft_reply
    # FIX: reads email then returns a draft prompt — no recursive agent.run().
    # The LLM in the ReAct loop writes the actual draft text from this context.
    # ------------------------------------------------------------------

    @tool
    async def draft_reply(
        message_id: str,
        instructions: str,
        tone: str = "professional",
    ) -> str:
        """
        Read an email and prepare context so you can write a draft reply.
        After calling this, write the draft reply yourself based on the email content.
        The draft is NOT sent — show it to the user and ask for confirmation.

        Args:
            message_id:   Gmail message ID of the email to reply to.
            instructions: What the reply should say or accomplish.
            tone:         Writing tone: professional, friendly, formal, brief.

        Returns:
            JSON with original email content + instructions for you to write the draft.
        """
        valid_tones = {"professional", "friendly", "formal", "brief"}
        tone = tone if tone in valid_tones else "professional"

        try:
            agent = await _agent()
            email = await agent.get_email(message_id)
        except ValueError as e:
            raise ToolException(str(e))
        except Exception as e:
            logger.error("tool.draft_reply.error", error=str(e), user_id=user_id)
            raise ToolException(f"Failed to read email for draft: {e}")

        return json.dumps({
            "action":       "write_draft",
            "tone":         tone,
            "instructions": instructions,
            "reply_to": {
                "message_id": email.message_id,
                "from":       email.sender,
                "subject":    email.subject,
                "date":       email.date,
                "body":       (email.body or email.snippet or "")[:3000],
            },
            "note": (
                "Based on the above email and instructions, write the draft reply now. "
                "Then show it to the user and ask: 'Shall I send this?'"
            ),
        })

    # ------------------------------------------------------------------
    # Tool 6: get_email_thread
    # ------------------------------------------------------------------

    @tool
    async def get_email_thread(thread_id: str) -> str:
        """
        Get all messages in an email thread for full conversation context.

        Args:
            thread_id: Gmail thread ID (from search_emails results).

        Returns:
            JSON with all messages in the thread in chronological order.
        """
        try:
            agent  = await _agent()
            emails = await agent.search_emails(
                query=f"thread:{thread_id}",
                max_results=20,
                include_body=True,
            )
        except ValueError as e:
            raise ToolException(str(e))
        except Exception as e:
            logger.error("tool.get_email_thread.error", error=str(e), user_id=user_id)
            raise ToolException(f"Failed to get thread: {e}")

        return json.dumps({
            "thread_id":     thread_id,
            "message_count": len(emails),
            "messages": [
                {
                    "message_id": e.message_id,
                    "from":       e.sender,
                    "date":       e.date,
                    "subject":    e.subject,
                    "body":       (e.body or e.snippet or "")[:2000],
                }
                for e in emails
            ],
        })

    return [
        search_emails,
        read_email,
        send_email,
        summarise_inbox,
        draft_reply,
        get_email_thread,
    ]

