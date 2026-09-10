"""app/services/agents/tools/task_tools.py
=========================================
LangChain @tool decorated functions for Task Agent database operations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from langchain_core.tools import tool
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.models import Task

logger = get_logger(__name__)


def build_task_tools(user_id: str, db: AsyncSession) -> list:
    """Factory that builds and returns tools bound to a specific user_id and DB session."""

    @tool
    async def create_task(
        title: str,
        description: str = "",
        priority: str = "medium",
        due_date: Optional[str] = None,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Create a new task for the user.

        Args:
            title: Short summary of the task.
            description: Detailed notes or instructions.
            priority: Priority level ('low', 'medium', 'high', 'urgent').
            due_date: Due date in YYYY-MM-DD format.
            tags: Optional list of tag strings.
        """
        try:
            task = Task(
                user_id=user_id,
                task_type="user_task",
                status="pending",
                input_data={
                    "title": title,
                    "description": description,
                    "priority": priority,
                    "due_date": due_date,
                    "tags": tags or [],
                },
            )
            db.add(task)
            await db.flush()
            await db.refresh(task)
            return json.dumps({"task_id": str(task.id), "title": title, "status": "created"})
        except Exception as e:
            logger.error("tool.create_task.error", error=str(e))
            return json.dumps({"error": f"Failed to create task: {str(e)}"})

    @tool
    async def list_tasks(
        status: str = "all",
        priority: str = "all",
        limit: int = 10,
    ) -> str:
        """List and filter tasks owned by the user.

        Args:
            status: Filter by status ('pending', 'running', 'success', 'failed', 'all').
            priority: Filter by priority ('low', 'medium', 'high', 'urgent', 'all').
            limit: Maximum number of records to return (default: 10).
        """
        try:
            q = select(Task).where(Task.user_id == user_id)
            if status != "all":
                q = q.where(Task.status == status)
            if priority != "all":
                q = q.where(Task.input_data["priority"].astext == priority)
            
            q = q.limit(limit).order_by(Task.created_at.desc())
            result = await db.execute(q)
            tasks = result.scalars().all()

            task_list = [
                {
                    "task_id": str(t.id),
                    "title": t.input_data.get("title", ""),
                    "description": t.input_data.get("description", ""),
                    "priority": t.input_data.get("priority", "medium"),
                    "due_date": t.input_data.get("due_date"),
                    "tags": t.input_data.get("tags", []),
                    "status": t.status,
                    "created_at": t.created_at.isoformat(),
                }
                for t in tasks
            ]
            return json.dumps({"tasks": task_list, "total": len(task_list)})
        except Exception as e:
            logger.error("tool.list_tasks.error", error=str(e))
            return json.dumps({"error": f"Failed to list tasks: {str(e)}"})

    @tool
    async def update_task(
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[str] = None,
        due_date: Optional[str] = None,
    ) -> str:
        """Update fields on an existing task by its UUID."""
        try:
            result = await db.execute(
                select(Task).where(Task.id == task_id, Task.user_id == user_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                return json.dumps({"error": f"Task {task_id} not found."})

            updated_data = dict(task.input_data)
            updates = {"title": title, "description": description, "priority": priority, "due_date": due_date}
            
            for field, val in updates.items():
                if val is not None:
                    updated_data[field] = val

            task.input_data = updated_data
            await db.flush()
            return json.dumps({"task_id": task_id, "status": "updated"})
        except Exception as e:
            logger.error("tool.update_task.error", error=str(e))
            return json.dumps({"error": f"Failed to update task: {str(e)}"})

    @tool
    async def complete_task(task_id: str) -> str:
        """Mark a specific task as completed."""
        try:
            result = await db.execute(
                select(Task).where(Task.id == task_id, Task.user_id == user_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                return json.dumps({"error": f"Task {task_id} not found."})

            task.status = "success"
            task.completed_at = datetime.now(timezone.utc)
            await db.flush()
            return json.dumps({"task_id": task_id, "status": "completed"})
        except Exception as e:
            logger.error("tool.complete_task.error", error=str(e))
            return json.dumps({"error": f"Failed to complete task: {str(e)}"})

    @tool
    async def delete_task(task_id: str) -> str:
        """Permanently delete a task by ID."""
        try:
            result = await db.execute(
                select(Task).where(Task.id == task_id, Task.user_id == user_id)
            )
            task = result.scalar_one_or_none()
            if not task:
                return json.dumps({"error": f"Task {task_id} not found."})

            await db.delete(task)
            await db.flush()
            return json.dumps({"task_id": task_id, "status": "deleted"})
        except Exception as e:
            logger.error("tool.delete_task.error", error=str(e))
            return json.dumps({"error": f"Failed to delete task: {str(e)}"})

    return [create_task, list_tasks, update_task, complete_task, delete_task]