"""
app/services/agents/calendar_agent.py
=======================================
Autonomous Google Calendar Agent powered by LangGraph ReAct loop.

AUTONOMOUS CAPABILITIES:
  The agent can:
    1. List and search calendar events across any date range
    2. Check which calendars the user has (primary, work, shared, etc.)
    3. Find free time slots for scheduling meetings
    4. Create events with attendees, location, Google Meet links
    5. Update existing events (reschedule, add attendees, rename)
    6. Delete events (always with confirmation)
    7. Handle natural language scheduling:
       "Schedule a 1-hour team standup every weekday at 9am"
       "Find a free 30-minute slot tomorrow afternoon"
       "Move my Friday 3pm meeting to next Monday same time"

SAFETY DESIGN (Write Operations):
  create_event, update_event, delete_event tool docstrings explicitly instruct
  the LLM to show details and ask for confirmation before executing.
  The system prompt reinforces this with CRITICAL SAFETY RULES.

OAUTH2 SCOPE:
  Requires: https://www.googleapis.com/auth/calendar
  Falls back with a clear error message if only Gmail scope was granted.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.agents.graph import build_agent_graph, run_agent
from app.services.agents.tools.calendar_tools import build_calendar_tools

settings = get_settings()
logger = get_logger(__name__)

CALENDAR_AGENT_SYSTEM_PROMPT = """You are an intelligent calendar assistant with direct access to the user's Google Calendar.

You can autonomously:
- View upcoming events and search by date range, keyword, or calendar
- List all calendars the user has access to
- Find free time slots for scheduling
- Create new events with full details (title, time, attendees, Meet link)
- Update or reschedule existing events
- Delete events

WRITE OPERATION RULES:
1. When asked to CREATE an event: show the full event details first, then ask
   "Shall I create this event?" — do NOT create without confirmation.
2. When asked to UPDATE an event: show current vs new details, then ask
   "Shall I update this event?" — do NOT update without confirmation.
3. When asked to DELETE an event: show the event details, then ask
   "Shall I delete this event?" — do NOT delete without confirmation.
4. When the Manager passes "User confirmed. Create/update/delete the event: [details]"
   — execute the calendar write operation immediately. Do NOT ask again.
5. Only skip confirmation when the instruction explicitly says "User confirmed".

READ OPERATIONS (no confirmation needed):
- Listing events, checking availability, finding free slots — execute immediately.

Always confirm timezone when scheduling across different regions.
Default to the user's local timezone if not specified

SCHEDULING WORKFLOW EXAMPLES:
  "What's on my calendar this week?":
    1. list_events(time_min=<monday>, time_max=<sunday>)
    2. Summarise the results clearly

  "Schedule a meeting with alice@example.com tomorrow at 2pm for 1 hour":
    1. list_events(time_min=<tomorrow 2pm>, time_max=<tomorrow 3pm>) — check for conflicts
    2. Show proposed event details to user
    3. [Only if confirmed] create_event(...)

  "Find a free 30-minute slot tomorrow afternoon":
    1. find_free_slots(date=<tomorrow>, duration_minutes=30, time_min="12:00", time_max="18:00")
    2. Present the options to the user and ask which they prefer

  "Move my 3pm Friday meeting to Monday":
    1. list_events(time_min=<friday 2:45pm>, time_max=<friday 3:15pm>) — find the event
    2. Show the proposed change: Friday 3pm → Monday 3pm
    3. [Only if confirmed] update_event(...)

DATETIME FORMAT:
  Always use ISO-8601 with UTC timezone: "2025-06-15T14:00:00Z"
  When users say relative times ("tomorrow", "next Monday", "in 2 hours"),
  calculate the actual datetime based on today's date before calling tools.
  Today's date is available in your context window.

RESPONSE STYLE:
  - Format event lists clearly with dates, times, and titles
  - Use human-friendly time formatting: "Monday 15 Jun, 2:00–3:00 PM"
  - For free slot lists, present as numbered options the user can pick from
  - Always confirm what action was taken after a successful write"""


class CalendarAgent:
    """
    LangGraph-powered autonomous Google Calendar agent.

    Handles scheduling, event management, and availability checking
    with autonomous multi-step reasoning and confirmation flows.
    """

    def __init__(self, user_id: str, db: AsyncSession) -> None:
        self.user_id = user_id
        self.db = db

        self.tools = build_calendar_tools(user_id=user_id, db=db)

        # llm = ChatAnthropic(
        #     model=settings.llm.primary_model,
        #     api_key=settings.llm.anthropic_api_key,
        #     temperature=0,
        #     max_tokens=4096,
        # )
        from app.api.v1.endpoints.chat import _build_llm
        llm = _build_llm(temperature=0, max_tokens=4096)
        self.llm_with_tools = llm.bind_tools(self.tools)
        self.graph = build_agent_graph(self.llm_with_tools, self.tools)

    async def run(
        self,
        instruction: str,
        conversation_history: list[BaseMessage] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Run the autonomous Calendar agent.

        Args:
            instruction:          Natural language calendar request.
            conversation_history: Prior messages for multi-turn context.

        Returns:
            (final_text_response, list_of_tool_calls_made)
        """
        return await run_agent(
            graph=self.graph,
            user_message=instruction,
            user_id=self.user_id,
            agent_type="calendar",
            system_prompt=CALENDAR_AGENT_SYSTEM_PROMPT,
            conversation_history=conversation_history,
        )
