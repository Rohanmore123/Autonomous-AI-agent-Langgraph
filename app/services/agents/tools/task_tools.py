"""
app/services/agents/tools/task_tools.py
=========================================
LangChain @tool decorated task management tools.

WHY @tool OVER RAW FUNCTIONS?
  The @tool decorator from langchain_core.tools does several things:
    1. Extracts the JSON schema from Python type hints → LLM sees a typed API
    2. Wraps the function in a BaseTool that LangGraph's ToolNode understands
    3. Catches exceptions and returns ToolException → agent can self-correct
    4. Enables LangSmith tracing when LANGCHAIN_API_KEY is set
    5. Supports args_schema validation (Pydantic) before calling the function

FACTORY PATTERN (build_task_tools):
  Tools need access to user_id and db (AsyncSession) at call time.
  We can't put async sessions in module-level globals (not thread-safe).
  Solution: a factory function that closes over user_id + db and returns
  a list of bound tool instances. Each request gets its own set of tools.

  Usage:
      tools = build_task_tools(user_id="u-123", db=session)
      agent = create_task_agent(tools)

ASYNC TOOLS:
  LangChain supports async tools natively (decorated with async def).
  LangGraph's ToolNode calls them with await when the graph runs async.

ERROR HANDLING:
  Tools raise ToolException on user-facing errors (task not found, etc.)
  The agent's error handling node catches this and lets the LLM decide
  whether to retry, ask for clarification, or report the error to the user.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import tool, ToolException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.models import Task

logger = get_logger(__name__)


def build_task_tools(user_id: str, db: AsyncSession) -> list:
    """
    Factory: create a set of task management tools bound to a specific
    user_id and DB session. Returns a list of LangChain BaseTool instances.
    """

    # ------------------------------------------------------------------
    # Tool 1: create_task
    # ------------------------------------------------------------------

    @tool
    async def create_task(
        title: str,
        description: str = "",
        priority: str = "medium",
        due_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """
        Create a new task for the user.

        Args:
            title:       Short, clear task title (required).
            description: Full task description with details.
            priority:    One of: low, medium, high, urgent.
            due_date:    Target completion date in ISO format (YYYY-MM-DD).
            tags:        List of tag strings for categorisation.

        Returns:
            JSON string confirming the created task with its ID.
        """
        # Validate priority
        valid_priorities = {"low", "medium", "high", "urgent"}
        if priority not in valid_priorities:
            raise ToolException(f"Invalid priority '{priority}'. Use: {valid_priorities}")

        task = Task(
            user_id=user_id,
            task_type="user_task",
            status="pending",
            input_data={
                "title":       title,
                "description": description,
                "priority":    priority,
                "due_date":    due_date,
                "tags":        tags or [],
            },
        )
        db.add(task)
        await db.flush()

        logger.info("tool.create_task", task_id=task.id, title=title, user_id=user_id)
        return json.dumps({
            "success":  True,
            "task_id":  task.id,
            "title":    title,
            "priority": priority,
            "due_date": due_date,
            "status":   "created",
        })

    # ------------------------------------------------------------------
    # Tool 2: list_tasks
    # ------------------------------------------------------------------

    @tool
    async def list_tasks(
        status: str = "all",
        priority: str = "all",
        limit: int = 10,
    ) -> str:
        """
        List the current user's tasks with optional filters.

        Args:
            status:   Filter by status. One of: pending, running, success, failed, all.
            priority: Filter by priority. One of: low, medium, high, urgent, all.
            limit:    Maximum number of tasks to return (1-50).

        Returns:
            JSON string with a list of tasks and their details.
        """
        limit = max(1, min(50, limit))

        q = (
            select(Task)
            .where(Task.user_id == user_id)
            .order_by(Task.created_at.desc())
            .limit(limit)
        )
        if status != "all":
            q = q.where(Task.status == status)

        result = await db.execute(q)
        tasks = result.scalars().all()

        # Priority filter happens in Python (JSONB query is complex cross-DB)
        if priority != "all":
            tasks = [t for t in tasks if t.input_data.get("priority") == priority]

        return json.dumps({
            "tasks": [
                {
                    "task_id":     t.id,
                    "title":       t.input_data.get("title", ""),
                    "description": t.input_data.get("description", ""),
                    "priority":    t.input_data.get("priority", "medium"),
                    "due_date":    t.input_data.get("due_date"),
                    "tags":        t.input_data.get("tags", []),
                    "status":      t.status,
                    "created_at":  t.created_at.isoformat(),
                    "completed_at": t.completed_at.isoformat() if t.completed_at else None,
                }
                for t in tasks
            ],
            "total": len(tasks),
            "filters": {"status": status, "priority": priority},
        })

    # ------------------------------------------------------------------
    # Tool 3: update_task
    # ------------------------------------------------------------------

    @tool
    async def update_task(
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        due_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """
        Update fields of an existing task. Only provide fields you want to change.

        Args:
            task_id:     UUID of the task to update (required).
            title:       New task title.
            description: New task description.
            priority:    New priority: low, medium, high, urgent.
            due_date:    New due date in ISO format (YYYY-MM-DD).
            tags:        New list of tags (replaces existing tags).

        Returns:
            JSON string confirming the update.
        """
        result = await db.execute(
            select(Task).where(Task.id == task_id, Task.user_id == user_id)
        )
        task = result.scalar_one_or_none()
        if not task:
            raise ToolException(f"Task '{task_id}' not found or you don't own it.")

        updated = dict(task.input_data)
        changed_fields = []

        for field, value in [
            ("title", title), ("description", description),
            ("priority", priority), ("due_date", due_date), ("tags", tags),
        ]:
            if value is not None:
                updated[field] = value
                changed_fields.append(field)

        task.input_data = updated
        await db.flush()

        logger.info("tool.update_task", task_id=task_id, fields=changed_fields)
        return json.dumps({
            "success":        True,
            "task_id":        task_id,
            "updated_fields": changed_fields,
            "current_values": {f: updated.get(f) for f in changed_fields},
        })

    # ------------------------------------------------------------------
    # Tool 4: complete_task
    # ------------------------------------------------------------------

    @tool
    async def complete_task(task_id: str) -> str:
        """
        Mark a task as completed.

        Args:
            task_id: UUID of the task to complete.

        Returns:
            JSON string confirming completion.
        """
        result = await db.execute(
            select(Task).where(Task.id == task_id, Task.user_id == user_id)
        )
        task = result.scalar_one_or_none()
        if not task:
            raise ToolException(f"Task '{task_id}' not found or you don't own it.")

        if task.status == "success":
            return json.dumps({"success": True, "task_id": task_id, "note": "Already completed."})

        task.status = "success"
        task.completed_at = datetime.now(timezone.utc)
        await db.flush()

        logger.info("tool.complete_task", task_id=task_id)
        return json.dumps({
            "success":      True,
            "task_id":      task_id,
            "title":        task.input_data.get("title", ""),
            "completed_at": task.completed_at.isoformat(),
        })

    # ------------------------------------------------------------------
    # Tool 5: delete_task
    # ------------------------------------------------------------------

    @tool
    async def delete_task(task_id: str) -> str:
        """
        Permanently delete a task.

        Args:
            task_id: UUID of the task to delete.

        Returns:
            JSON string confirming deletion.
        """
        result = await db.execute(
            select(Task).where(Task.id == task_id, Task.user_id == user_id)
        )
        task = result.scalar_one_or_none()
        if not task:
            raise ToolException(f"Task '{task_id}' not found or you don't own it.")

        title = task.input_data.get("title", "")
        await db.delete(task)
        await db.flush()

        logger.info("tool.delete_task", task_id=task_id)
        return json.dumps({"success": True, "task_id": task_id, "deleted_title": title})

    # ------------------------------------------------------------------
    # Tool 6: get_task_summary
    # ------------------------------------------------------------------

    @tool
    async def get_task_summary() -> str:
        """
        Get a summary of the user's task statistics (count by status and priority).

        Returns:
            JSON string with task counts broken down by status and priority.
        """
        result = await db.execute(
            select(Task).where(Task.user_id == user_id)
        )
        all_tasks = result.scalars().all()

        by_status: dict[str, int] = {}
        by_priority: dict[str, int] = {}
        overdue: list[dict] = []
        today = datetime.now(timezone.utc).date()

        for t in all_tasks:
            by_status[t.status] = by_status.get(t.status, 0) + 1
            p = t.input_data.get("priority", "medium")
            by_priority[p] = by_priority.get(p, 0) + 1

            due = t.input_data.get("due_date")
            if due and t.status == "pending":
                try:
                    from datetime import date
                    due_date = date.fromisoformat(due)
                    if due_date < today:
                        overdue.append({"task_id": t.id, "title": t.input_data.get("title", "")})
                except ValueError:
                    pass

        return json.dumps({
            "total":       len(all_tasks),
            "by_status":   by_status,
            "by_priority": by_priority,
            "overdue":     overdue,
            "overdue_count": len(overdue),
        })

    return [create_task, list_tasks, update_task, complete_task, delete_task, get_task_summary]