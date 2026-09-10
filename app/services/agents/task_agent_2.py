"""
app/services/agents/task_agent.py
===================================
Task Management Agent — lets users create, list, update, and complete tasks
using natural language, powered by LLM tool-use (function calling).

TOOL-USE PATTERN (Agentic Loop):
  1. User sends: "Create a task to review the Q3 report by Friday, high priority"
  2. We send this + tool definitions to the LLM
  3. LLM responds with: tool_use block → {name: "create_task", input: {...}}
  4. We execute the tool (DB write)
  5. We send the tool result back to the LLM
  6. LLM generates final human response: "Task created: ..."

This is an agentic loop — the LLM decides which tools to call, we execute them,
and the LLM synthesizes the final answer.  The loop runs until the LLM stops
calling tools (returns a text-only response).

MAX_ITERATIONS = 5 prevents infinite loops from misbehaving models.

TOOLS AVAILABLE:
  create_task      — add a new task to the DB
  list_tasks       — retrieve tasks by status/priority
  update_task      — change title, description, priority, due_date
  complete_task    — mark as done
  delete_task      — remove a task
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import anthropic
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.models import Task

settings = get_settings()
logger = get_logger(__name__)

MAX_ITERATIONS = 5

# ---------------------------------------------------------------------------
# Tool schemas (Anthropic tool_use format)
# ---------------------------------------------------------------------------

TASK_TOOLS: list[dict] = [
    {
        "name": "create_task",
        "description": "Create a new task for the user",
        "input_schema": {
            "type": "object",
            "properties": {
                "title":       {"type": "string", "description": "Short task title"},
                "description": {"type": "string", "description": "Full task description"},
                "priority":    {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "due_date":    {"type": "string", "description": "ISO date string, e.g. 2025-12-31"},
                "tags":        {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title"],
        },
    },
    {
        "name": "list_tasks",
        "description": "List the user's tasks, optionally filtered by status or priority",
        "input_schema": {
            "type": "object",
            "properties": {
                "status":   {"type": "string", "enum": ["pending", "running", "success", "failed", "all"]},
                "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent", "all"]},
                "limit":    {"type": "integer", "minimum": 1, "maximum": 50, "default": 10},
            },
        },
    },
    {
        "name": "update_task",
        "description": "Update fields of an existing task by its ID",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id":     {"type": "string", "description": "Task UUID"},
                "title":       {"type": "string"},
                "description": {"type": "string"},
                "priority":    {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
                "due_date":    {"type": "string"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "complete_task",
        "description": "Mark a task as completed",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task UUID"},
            },
            "required": ["task_id"],
        },
    },
    {
        "name": "delete_task",
        "description": "Permanently delete a task",
        "input_schema": {
            "type": "object",
            "properties": {
                "task_id": {"type": "string", "description": "Task UUID"},
            },
            "required": ["task_id"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

class TaskToolExecutor:
    """Executes task management tools against the PostgreSQL database."""

    def __init__(self, user_id: str, db: AsyncSession) -> None:
        self.user_id = user_id
        self.db = db

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        """Dispatch tool call to the right handler. Returns JSON string result."""
        handlers = {
            "create_task":   self._create_task,
            "list_tasks":    self._list_tasks,
            "update_task":   self._update_task,
            "complete_task": self._complete_task,
            "delete_task":   self._delete_task,
        }
        handler = handlers.get(tool_name)
        if not handler:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

        try:
            result = await handler(**tool_input)
            return json.dumps(result)
        except Exception as exc:
            logger.error("task_agent.tool.error", tool=tool_name, error=str(exc))
            return json.dumps({"error": str(exc)})

    async def _create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_date: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        task = Task(
            user_id=self.user_id,
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
        self.db.add(task)
        await self.db.flush()
        await self.db.refresh(task)
        return {"task_id": task.id, "title": title, "status": "created"}

    async def _list_tasks(
        self,
        status: str = "all",
        priority: str = "all",
        limit: int = 10,
    ) -> dict:
        q = select(Task).where(Task.user_id == self.user_id)
        if status != "all":
            q = q.where(Task.status == status)
        if priority != "all":
            q = q.where(Task.input_data["priority"].astext == priority)
        q = q.limit(limit).order_by(Task.created_at.desc())

        result = await self.db.execute(q)
        tasks = result.scalars().all()

        return {
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
                }
                for t in tasks
            ],
            "total": len(tasks),
        }

    async def _update_task(self, task_id: str, **fields) -> dict:
        result = await self.db.execute(
            select(Task).where(Task.id == task_id, Task.user_id == self.user_id)
        )
        task = result.scalar_one_or_none()
        if not task:
            return {"error": f"Task {task_id} not found"}

        # Merge updates into input_data JSONB
        updated_data = dict(task.input_data)
        for k, v in fields.items():
            if v is not None:
                updated_data[k] = v
        task.input_data = updated_data
        await self.db.flush()
        return {"task_id": task_id, "status": "updated", "fields": list(fields.keys())}

    async def _complete_task(self, task_id: str) -> dict:
        result = await self.db.execute(
            select(Task).where(Task.id == task_id, Task.user_id == self.user_id)
        )
        task = result.scalar_one_or_none()
        if not task:
            return {"error": f"Task {task_id} not found"}

        task.status = "success"
        task.completed_at = datetime.now(timezone.utc)
        await self.db.flush()
        return {"task_id": task_id, "status": "completed"}

    async def _delete_task(self, task_id: str) -> dict:
        result = await self.db.execute(
            select(Task).where(Task.id == task_id, Task.user_id == self.user_id)
        )
        task = result.scalar_one_or_none()
        if not task:
            return {"error": f"Task {task_id} not found"}

        await self.db.delete(task)
        await self.db.flush()
        return {"task_id": task_id, "status": "deleted"}


# ---------------------------------------------------------------------------
# Task Agent
# ---------------------------------------------------------------------------

class TaskAgent:
    """
    Agentic task manager using Anthropic tool_use.

    The agent loops until:
      a) The LLM returns a final text response (stop_reason == "end_turn")
      b) MAX_ITERATIONS reached (safety guard)
    """

    def __init__(self, user_id: str, db: AsyncSession) -> None:
        self.user_id = user_id
        self.db = db
        self.executor = TaskToolExecutor(user_id, db)
        self._client = anthropic.AsyncAnthropic(api_key=settings.llm.anthropic_api_key)

    async def run(
        self,
        instruction: str,
        context: dict[str, Any] | None = None,
    ) -> tuple[str, list[dict]]:
        """
        Run the agentic loop.
        Returns: (final_text_response, list_of_tool_calls_made)
        """
        system = (
            "You are a helpful task management assistant. "
            "Use the provided tools to manage the user's tasks. "
            "Always confirm what action you took and summarise the result."
        )

        messages: list[dict] = [{"role": "user", "content": instruction}]
        if context:
            messages[0]["content"] += f"\n\nContext: {json.dumps(context)}"

        tool_calls_made: list[dict] = []
        final_response = ""

        for iteration in range(MAX_ITERATIONS):
            logger.info("task_agent.iteration", iteration=iteration, user_id=self.user_id)

            response = await self._client.messages.create(
                model=settings.llm.primary_model,
                max_tokens=1024,
                system=system,
                tools=TASK_TOOLS,
                messages=messages,
            )

            # Add assistant response to message history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                # LLM finished — extract text response
                for block in response.content:
                    if hasattr(block, "text"):
                        final_response = block.text
                        break
                break

            if response.stop_reason == "tool_use":
                # Execute all tool calls in this response
                tool_results = []

                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input

                        logger.info(
                            "task_agent.tool_call",
                            tool=tool_name,
                            input=str(tool_input)[:100],
                        )

                        result_str = await self.executor.execute(tool_name, tool_input)
                        tool_calls_made.append(
                            {"tool": tool_name, "input": tool_input, "result": result_str}
                        )

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                            }
                        )

                # Feed tool results back to the LLM
                messages.append({"role": "user", "content": tool_results})

            else:
                # Unexpected stop reason
                logger.warning("task_agent.unexpected_stop", reason=response.stop_reason)
                break

        if not final_response:
            final_response = "Task operation completed."

        # Commit all DB changes made by tools
        await self.db.commit()

        logger.info(
            "task_agent.complete",
            tool_calls=len(tool_calls_made),
            user_id=self.user_id,
        )

        return final_response, tool_calls_made