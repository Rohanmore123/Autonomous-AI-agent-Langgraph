"""
app/services/agents/tools/calendar_tools.py
=============================================
LangChain @tool decorated Google Calendar tools.

TOOLS:
  list_events        — fetch upcoming events in a time range
  get_event          — get full details of a specific event
  create_event       — create a new calendar event
  update_event       — update fields of an existing event
  delete_event       — delete an event
  find_free_slots    — find available time gaps between events
  list_calendars     — list all calendars the user has access to

OAUTH2:
  Reuses the GoogleOAuthToken row created by the Gmail OAuth flow.
  Requires the calendar scope: https://www.googleapis.com/auth/calendar
  If only the Gmail scope was granted, tools raise a clear ToolException
  asking the user to re-authenticate with calendar permissions.

SAFETY DESIGN:
  create_event and update_event always return a preview of what will be
  written. The LangGraph agent is instructed (via calendar_agent.py system
  prompt) to show this preview and ask for confirmation before writing.
  delete_event requires the event ID — it will not bulk-delete.

TIMEZONE:
  All datetimes are stored and returned in UTC ISO-8601 format.
  The agent converts to/from the user's local timezone using the
  calendar's configured timezone (fetched from the Calendar API).

RATE LIMITING:
  Google Calendar API quota: 1,000,000 queries/day, 500/100s per user.
  Well within normal usage — no explicit back-off needed unless bulk ops.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from typing import Optional

from langchain_core.tools import tool, ToolException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.logging import get_logger

settings = get_settings()
logger = get_logger(__name__)

CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"


def build_calendar_tools(user_id: str, db: AsyncSession) -> list:
    """
    Factory: create Google Calendar tools bound to user_id and DB session.
    Credentials are loaded from the GoogleOAuthToken row (same as Gmail).
    """

    # ------------------------------------------------------------------
    # Internal helper: build an authenticated Google Calendar service
    # ------------------------------------------------------------------

    async def _get_calendar_service():
        """
        Load and refresh OAuth2 credentials, then return a Google Calendar
        API service object. Raises ToolException if no token or wrong scope.
        """
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from sqlalchemy import select
        from app.models.models import GoogleOAuthToken

        result = await db.execute(
            select(GoogleOAuthToken).where(GoogleOAuthToken.user_id == user_id)
        )
        token_row = result.scalar_one_or_none()
        if not token_row:
            raise ToolException(
                "Google account not connected. "
                "Please authenticate via /api/v1/auth/google to use Calendar features."
            )

        # Check that the calendar scope was granted
        granted_scopes = token_row.scopes or []
        if CALENDAR_SCOPE not in granted_scopes:
            raise ToolException(
                "Calendar access not granted. "
                "Please re-authenticate at /api/v1/auth/google and allow Calendar permissions."
            )

        creds = Credentials(
            token=token_row.access_token,
            refresh_token=token_row.refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_oauth.client_id,
            client_secret=settings.google_oauth.client_secret,
            scopes=token_row.scopes,
        )

        if creds.expired and creds.refresh_token:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, lambda: creds.refresh(Request()))
            token_row.access_token = creds.token
            if creds.expiry:
                token_row.token_expiry = creds.expiry.replace(tzinfo=timezone.utc)
            await db.commit()

        loop = asyncio.get_event_loop()
        service = await loop.run_in_executor(
            None,
            lambda: build("calendar", "v3", credentials=creds, cache_discovery=False),
        )
        return service

    def _parse_event(event: dict) -> dict:
        """Normalise a raw Google Calendar event dict into a clean schema."""
        start = event.get("start", {})
        end = event.get("end", {})
        return {
            "event_id":    event.get("id", ""),
            "title":       event.get("summary", "(no title)"),
            "description": event.get("description", ""),
            "location":    event.get("location", ""),
            "start":       start.get("dateTime") or start.get("date", ""),
            "end":         end.get("dateTime") or end.get("date", ""),
            "all_day":     "date" in start and "dateTime" not in start,
            "calendar_id": event.get("organizer", {}).get("email", "primary"),
            "attendees": [
                {
                    "email":        a.get("email", ""),
                    "name":         a.get("displayName", ""),
                    "response":     a.get("responseStatus", ""),
                }
                for a in event.get("attendees", [])
            ],
            "status":      event.get("status", "confirmed"),
            "meet_link":   event.get("hangoutLink", ""),
            "html_link":   event.get("htmlLink", ""),
            "recurrence":  event.get("recurrence", []),
        }

    # ------------------------------------------------------------------
    # Tool 1: list_events
    # ------------------------------------------------------------------

    @tool
    async def list_events(
        time_min: Optional[str] = None,
        time_max: Optional[str] = None,
        max_results: int = 10,
        calendar_id: str = "primary",
        query: Optional[str] = None,
    ) -> str:
        """
        List upcoming calendar events in a time range.

        Args:
            time_min:    Start of range in ISO-8601 format (default: now).
                         Examples: "2025-06-01T00:00:00Z", "2025-06-01" (date only).
            time_max:    End of range in ISO-8601 format (default: 7 days from now).
            max_results: Max events to return (1-50, default 10).
            calendar_id: Calendar to query. Use "primary" for the main calendar,
                         or a calendar ID from list_calendars.
            query:       Free-text search within event titles and descriptions.

        Returns:
            JSON string with list of events including title, time, attendees, location.
        """
        max_results = max(1, min(50, max_results))

        now = datetime.now(timezone.utc)
        if not time_min:
            time_min = now.isoformat()
        if not time_max:
            time_max = (now + timedelta(days=7)).isoformat()

        # Ensure timezone suffix
        if time_min and not time_min.endswith("Z") and "+" not in time_min:
            time_min += "Z"
        if time_max and not time_max.endswith("Z") and "+" not in time_max:
            time_max += "Z"

        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()

            def _sync():
                kwargs = dict(
                    calendarId=calendar_id,
                    timeMin=time_min,
                    timeMax=time_max,
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy="startTime",
                )
                if query:
                    kwargs["q"] = query
                return service.events().list(**kwargs).execute()

            result = await loop.run_in_executor(None, _sync)
            events = result.get("items", [])
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to list events: {str(e)}")

        if not events:
            return json.dumps({
                "found": 0,
                "events": [],
                "message": "No events found in this time range.",
                "time_min": time_min,
                "time_max": time_max,
            })

        return json.dumps({
            "found":    len(events),
            "time_min": time_min,
            "time_max": time_max,
            "events":   [_parse_event(e) for e in events],
        })

    # ------------------------------------------------------------------
    # Tool 2: get_event
    # ------------------------------------------------------------------

    @tool
    async def get_event(event_id: str, calendar_id: str = "primary") -> str:
        """
        Get full details of a specific calendar event by its ID.
        Use event IDs from list_events results.

        Args:
            event_id:    Google Calendar event ID.
            calendar_id: Calendar containing the event (default: "primary").

        Returns:
            JSON string with complete event details including attendees and links.
        """
        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()
            event = await loop.run_in_executor(
                None,
                lambda: service.events().get(
                    calendarId=calendar_id, eventId=event_id
                ).execute(),
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to get event '{event_id}': {str(e)}")

        return json.dumps(_parse_event(event))

    # ------------------------------------------------------------------
    # Tool 3: create_event
    # ------------------------------------------------------------------

    @tool
    async def create_event(
        title: str,
        start: str,
        end: str,
        description: str = "",
        location: str = "",
        attendees: Optional[list[str]] = None,
        calendar_id: str = "primary",
        add_google_meet: bool = False,
        all_day: bool = False,
    ) -> str:
        """
        Create a new calendar event.

        IMPORTANT: Always show the event details to the user and ask for
        confirmation before calling this tool. Say "shall I create this event?"

        Args:
            title:          Event title / summary (required).
            start:          Start datetime in ISO-8601 (e.g. "2025-06-15T14:00:00Z").
                            For all-day events use date only: "2025-06-15".
            end:            End datetime in ISO-8601 (e.g. "2025-06-15T15:00:00Z").
                            For all-day events use date only: "2025-06-16" (exclusive).
            description:    Event description or notes.
            location:       Physical or virtual location string.
            attendees:      List of attendee email addresses to invite.
            calendar_id:    Target calendar (default: "primary").
            add_google_meet: Whether to add a Google Meet video link.
            all_day:        Whether this is an all-day event (uses date, not dateTime).

        Returns:
            JSON string confirming the created event with its ID and HTML link.
        """
        if not title.strip():
            raise ToolException("Event title cannot be empty.")
        if not start or not end:
            raise ToolException("Both start and end datetimes are required.")

        # Build the event body
        if all_day:
            time_spec = lambda dt: {"date": dt.split("T")[0]}
        else:
            def time_spec(dt):
                if not dt.endswith("Z") and "+" not in dt:
                    dt += "Z"
                return {"dateTime": dt, "timeZone": "UTC"}

        event_body: dict = {
            "summary":     title,
            "description": description,
            "location":    location,
            "start":       time_spec(start),
            "end":         time_spec(end),
        }

        if attendees:
            event_body["attendees"] = [{"email": email} for email in attendees]

        if add_google_meet:
            event_body["conferenceData"] = {
                "createRequest": {
                    "requestId": f"meet-{user_id}-{int(datetime.now().timestamp())}",
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            }

        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()

            def _sync():
                kwargs = dict(calendarId=calendar_id, body=event_body)
                if add_google_meet:
                    kwargs["conferenceDataVersion"] = 1
                return service.events().insert(**kwargs).execute()

            created = await loop.run_in_executor(None, _sync)
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to create event: {str(e)}")

        logger.info(
            "tool.create_event",
            user_id=user_id,
            event_id=created.get("id"),
            title=title,
        )

        parsed = _parse_event(created)
        return json.dumps({
            "success":   True,
            "action":    "created",
            "event":     parsed,
            "html_link": created.get("htmlLink", ""),
            "meet_link": created.get("hangoutLink", ""),
        })

    # ------------------------------------------------------------------
    # Tool 4: update_event
    # ------------------------------------------------------------------

    @tool
    async def update_event(
        event_id: str,
        title: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        description: Optional[str] = None,
        location: Optional[str] = None,
        attendees_add: Optional[list[str]] = None,
        attendees_remove: Optional[list[str]] = None,
        calendar_id: str = "primary",
    ) -> str:
        """
        Update fields of an existing calendar event. Only provide fields to change.

        IMPORTANT: Show the proposed changes to the user and confirm before calling.

        Args:
            event_id:         Google Calendar event ID (from list_events).
            title:            New event title.
            start:            New start datetime in ISO-8601.
            end:              New end datetime in ISO-8601.
            description:      New description.
            location:         New location.
            attendees_add:    List of email addresses to add as attendees.
            attendees_remove: List of email addresses to remove from attendees.
            calendar_id:      Calendar containing the event.

        Returns:
            JSON string confirming the updated event.
        """
        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()

            # Fetch current event to patch
            current = await loop.run_in_executor(
                None,
                lambda: service.events().get(
                    calendarId=calendar_id, eventId=event_id
                ).execute(),
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Event '{event_id}' not found: {str(e)}")

        # Apply changes
        patch: dict = {}
        if title is not None:
            patch["summary"] = title
        if description is not None:
            patch["description"] = description
        if location is not None:
            patch["location"] = location
        if start is not None:
            if not start.endswith("Z") and "+" not in start and "T" in start:
                start += "Z"
            patch["start"] = {"dateTime": start, "timeZone": "UTC"}
        if end is not None:
            if not end.endswith("Z") and "+" not in end and "T" in end:
                end += "Z"
            patch["end"] = {"dateTime": end, "timeZone": "UTC"}

        # Manage attendees
        if attendees_add or attendees_remove:
            current_attendees = {
                a["email"]: a for a in current.get("attendees", [])
            }
            if attendees_add:
                for email in attendees_add:
                    current_attendees[email] = {"email": email}
            if attendees_remove:
                for email in attendees_remove:
                    current_attendees.pop(email, None)
            patch["attendees"] = list(current_attendees.values())

        if not patch:
            return json.dumps({"success": True, "note": "No changes specified."})

        try:
            updated = await loop.run_in_executor(
                None,
                lambda: service.events().patch(
                    calendarId=calendar_id, eventId=event_id, body=patch
                ).execute(),
            )
        except Exception as e:
            raise ToolException(f"Failed to update event: {str(e)}")

        logger.info("tool.update_event", user_id=user_id, event_id=event_id)
        return json.dumps({
            "success":        True,
            "action":         "updated",
            "event":          _parse_event(updated),
            "updated_fields": list(patch.keys()),
        })

    # ------------------------------------------------------------------
    # Tool 5: delete_event
    # ------------------------------------------------------------------

    @tool
    async def delete_event(event_id: str, calendar_id: str = "primary") -> str:
        """
        Permanently delete a calendar event.

        IMPORTANT: Always confirm with the user before deleting. Show the
        event title and time, then ask "shall I delete this event?"

        Args:
            event_id:    Google Calendar event ID to delete.
            calendar_id: Calendar containing the event (default: "primary").

        Returns:
            JSON string confirming deletion.
        """
        # Fetch title first so we can confirm in the response
        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()

            event = await loop.run_in_executor(
                None,
                lambda: service.events().get(
                    calendarId=calendar_id, eventId=event_id
                ).execute(),
            )
            title = event.get("summary", "(no title)")
            start = (event.get("start", {}).get("dateTime")
                     or event.get("start", {}).get("date", ""))

            await loop.run_in_executor(
                None,
                lambda: service.events().delete(
                    calendarId=calendar_id, eventId=event_id
                ).execute(),
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to delete event '{event_id}': {str(e)}")

        logger.info("tool.delete_event", user_id=user_id, event_id=event_id, title=title)
        return json.dumps({
            "success":       True,
            "action":        "deleted",
            "event_id":      event_id,
            "deleted_title": title,
            "deleted_start": start,
        })

    # ------------------------------------------------------------------
    # Tool 6: find_free_slots
    # ------------------------------------------------------------------

    @tool
    async def find_free_slots(
        date: str,
        duration_minutes: int = 60,
        time_min: str = "09:00",
        time_max: str = "18:00",
        calendar_ids: Optional[list[str]] = None,
    ) -> str:
        """
        Find available (free) time slots on a given day.
        Uses Google Calendar's freebusy API to check all specified calendars.

        Args:
            date:             Date to check in YYYY-MM-DD format (e.g. "2025-06-15").
            duration_minutes: Required meeting length in minutes (default 60).
            time_min:         Earliest acceptable start time in HH:MM (24h, default "09:00").
            time_max:         Latest acceptable end time in HH:MM (24h, default "18:00").
            calendar_ids:     List of calendar IDs to check (default: ["primary"]).
                              Use list_calendars to get all calendar IDs.

        Returns:
            JSON string with list of free time windows of the requested duration.
        """
        calendar_ids = calendar_ids or ["primary"]
        duration_minutes = max(15, duration_minutes)

        # Build time bounds for the day
        try:
            day_start = datetime.fromisoformat(f"{date}T{time_min}:00").replace(tzinfo=timezone.utc)
            day_end   = datetime.fromisoformat(f"{date}T{time_max}:00").replace(tzinfo=timezone.utc)
        except ValueError as e:
            raise ToolException(f"Invalid date or time format: {e}")

        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()

            freebusy_body = {
                "timeMin": day_start.isoformat(),
                "timeMax": day_end.isoformat(),
                "items":   [{"id": cid} for cid in calendar_ids],
            }

            fb_result = await loop.run_in_executor(
                None,
                lambda: service.freebusy().query(body=freebusy_body).execute(),
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to query freebusy: {str(e)}")

        # Collect and merge all busy windows across calendars
        busy_windows: list[tuple[datetime, datetime]] = []
        for cal_data in fb_result.get("calendars", {}).values():
            for busy in cal_data.get("busy", []):
                b_start = datetime.fromisoformat(
                    busy["start"].replace("Z", "+00:00")
                )
                b_end = datetime.fromisoformat(
                    busy["end"].replace("Z", "+00:00")
                )
                busy_windows.append((b_start, b_end))

        # Sort and merge overlapping busy windows
        busy_windows.sort(key=lambda x: x[0])
        merged: list[tuple[datetime, datetime]] = []
        for b in busy_windows:
            if merged and b[0] < merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], b[1]))
            else:
                merged.append(b)

        # Find free slots of the required duration
        free_slots: list[dict] = []
        cursor = day_start
        slot_duration = timedelta(minutes=duration_minutes)

        for busy_start, busy_end in merged:
            if cursor + slot_duration <= busy_start:
                # There's a free gap before this busy block
                slot_end = cursor + slot_duration
                while slot_end <= busy_start:
                    free_slots.append({
                        "start":            cursor.strftime("%H:%M"),
                        "end":              slot_end.strftime("%H:%M"),
                        "start_iso":        cursor.isoformat(),
                        "end_iso":          slot_end.isoformat(),
                        "duration_minutes": duration_minutes,
                    })
                    cursor = slot_end
                    slot_end = cursor + slot_duration
            cursor = max(cursor, busy_end)

        # Check gap after last busy block
        slot_end = cursor + slot_duration
        while slot_end <= day_end:
            free_slots.append({
                "start":            cursor.strftime("%H:%M"),
                "end":              slot_end.strftime("%H:%M"),
                "start_iso":        cursor.isoformat(),
                "end_iso":          slot_end.isoformat(),
                "duration_minutes": duration_minutes,
            })
            cursor = slot_end
            slot_end = cursor + slot_duration

        return json.dumps({
            "date":             date,
            "duration_minutes": duration_minutes,
            "working_hours":    f"{time_min}–{time_max}",
            "free_slots_found": len(free_slots),
            "free_slots":       free_slots,
            "busy_count":       len(merged),
            "message": (
                f"Found {len(free_slots)} free slot(s) of {duration_minutes} minutes."
                if free_slots else
                f"No free {duration_minutes}-minute slots on {date} between {time_min}–{time_max}."
            ),
        })

    # ------------------------------------------------------------------
    # Tool 7: list_calendars
    # ------------------------------------------------------------------

    @tool
    async def list_calendars() -> str:
        """
        List all calendars the user has access to (own calendars + shared).
        Use this to discover calendar IDs before querying specific calendars,
        or to check which calendars are visible.

        Returns:
            JSON string with calendar IDs, names, and access roles.
        """
        try:
            service = await _get_calendar_service()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: service.calendarList().list().execute(),
            )
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Failed to list calendars: {str(e)}")

        calendars = result.get("items", [])
        return json.dumps({
            "total": len(calendars),
            "calendars": [
                {
                    "calendar_id":   c.get("id", ""),
                    "name":          c.get("summary", ""),
                    "description":   c.get("description", ""),
                    "timezone":      c.get("timeZone", ""),
                    "access_role":   c.get("accessRole", ""),
                    "primary":       c.get("primary", False),
                    "background_color": c.get("backgroundColor", ""),
                }
                for c in calendars
            ],
        })

    return [
        list_events,
        get_event,
        create_event,
        update_event,
        delete_event,
        find_free_slots,
        list_calendars,
    ]
