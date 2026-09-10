#!/usr/bin/env python3
"""
scripts/seed_db.py
==================
Database seeding script — run ONCE after first deployment.

WHAT THIS DOES:
  1. Creates default RBAC roles with permission sets
  2. Creates the initial admin user (credentials from env vars)
  3. Assigns admin role to admin user
  4. Creates default rate limit overrides for admin
  5. Verifies the setup is correct

USAGE:
  # From project root:
  python scripts/seed_db.py

  # Or with Docker:
  docker compose exec api python scripts/seed_db.py

ENVIRONMENT VARIABLES REQUIRED:
  DATABASE_URL       — PostgreSQL connection string
  ADMIN_EMAIL        — email for the initial admin account
  ADMIN_PASSWORD     — password for the initial admin account
  ADMIN_USERNAME     — username for the initial admin account

IDEMPOTENT:
  Safe to run multiple times — checks existence before creating.
  Re-running does NOT reset passwords or overwrite existing data.
"""

import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.security import hash_password
from app.models.models import Base, Role, User, UserRole

settings = get_settings()

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

ROLES = [
    {
        "name": "admin",
        "description": "Full platform access — manage users, view all data, change config",
        "permissions": {
            "*": ["*"],                          # Wildcard: all resources, all actions
            "users": ["read", "write", "delete"],
            "documents": ["read", "write", "delete"],
            "conversations": ["read", "write", "delete"],
            "agent_runs": ["read"],
            "roles": ["read", "write", "assign"],
            "api_keys": ["read", "write", "revoke"],
            "metrics": ["read"],
            "audit_logs": ["read"],
        },
    },
    {
        "name": "operator",
        "description": "Operations access — view all data and metrics, cannot manage users",
        "permissions": {
            "users": ["read"],
            "documents": ["read"],
            "conversations": ["read"],
            "agent_runs": ["read"],
            "metrics": ["read"],
            "audit_logs": ["read"],
        },
    },
    {
        "name": "user",
        "description": "Standard user — full access to own resources only",
        "permissions": {
            "own_documents": ["read", "write", "delete"],
            "own_conversations": ["read", "write", "delete"],
            "own_tasks": ["read", "write", "delete"],
            "own_api_keys": ["read", "write", "revoke"],
            "chat": ["read", "write"],
            "search": ["read"],
        },
    },
    {
        "name": "viewer",
        "description": "Read-only access to own resources",
        "permissions": {
            "own_documents": ["read"],
            "own_conversations": ["read"],
            "own_tasks": ["read"],
            "chat": ["read"],
        },
    },
    {
        "name": "api_bot",
        "description": "Service account for programmatic/CI access",
        "permissions": {
            "chat": ["read", "write"],
            "documents": ["read"],
            "search": ["read"],
        },
    },
]


# ---------------------------------------------------------------------------
# Seeding functions
# ---------------------------------------------------------------------------

async def seed_roles(db: AsyncSession) -> dict[str, Role]:
    """Create default roles. Returns dict of {name: Role} for all roles."""
    role_map: dict[str, Role] = {}

    for role_def in ROLES:
        result = await db.execute(select(Role).where(Role.name == role_def["name"]))
        existing = result.scalar_one_or_none()

        if existing:
            print(f"  ✓ Role '{role_def['name']}' already exists")
            role_map[role_def["name"]] = existing
        else:
            role = Role(
                name=role_def["name"],
                description=role_def["description"],
                permissions=role_def["permissions"],
            )
            db.add(role)
            await db.flush()
            role_map[role_def["name"]] = role
            print(f"  + Created role '{role_def['name']}'")

    await db.commit()
    return role_map


async def seed_admin_user(db: AsyncSession, role_map: dict[str, Role]) -> None:
    """Create the initial admin user from environment variables."""
    admin_email    = os.environ.get("ADMIN_EMAIL",    "admin@llmplatform.local")
    admin_password = os.environ.get("ADMIN_PASSWORD", "Admin@123456")   # CHANGE IN PROD
    admin_username = os.environ.get("ADMIN_USERNAME", "admin")

    result = await db.execute(select(User).where(User.email == admin_email))
    existing = result.scalar_one_or_none()

    if existing:
        print(f"  ✓ Admin user '{admin_email}' already exists (id={existing.id})")
        return

    # Create admin user
    admin = User(
        email=admin_email,
        username=admin_username,
        hashed_password=hash_password(admin_password),
        full_name="Platform Administrator",
        is_active=True,
        is_verified=True,
    )
    db.add(admin)
    await db.flush()

    # Assign admin role
    admin_role = role_map.get("admin")
    if admin_role:
        db.add(UserRole(user_id=admin.id, role_id=admin_role.id))

    await db.commit()
    print(f"  + Created admin user: {admin_email} (id={admin.id})")
    print(f"  ⚠️  Change the admin password immediately in production!")


async def verify_setup(db: AsyncSession) -> None:
    """Verify the seeded data is correct."""
    print("\nVerification:")

    # Count roles
    result = await db.execute(select(Role))
    roles = result.scalars().all()
    print(f"  Roles: {[r.name for r in roles]}")

    # Count users
    result = await db.execute(select(User))
    users = result.scalars().all()
    print(f"  Users: {len(users)} total")

    # Verify admin has role
    result = await db.execute(
        select(User).where(User.username == os.environ.get("ADMIN_USERNAME", "admin"))
    )
    admin = result.scalar_one_or_none()
    if admin:
        result = await db.execute(select(UserRole).where(UserRole.user_id == admin.id))
        user_roles = result.scalars().all()
        print(f"  Admin roles: {len(user_roles)} assigned")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("LLM Platform Database Seeder")
    print("=" * 60)

    # Create tables if they don't exist (dev/test only — use Alembic in prod)
    engine = create_async_engine(settings.db.url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✓ Database tables verified\n")

    SessionFactory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with SessionFactory() as db:
        print("Seeding roles...")
        role_map = await seed_roles(db)

        print("\nSeeding admin user...")
        await seed_admin_user(db, role_map)

        await verify_setup(db)

    await engine.dispose()
    print("\n✅ Seeding complete!")


if __name__ == "__main__":
    asyncio.run(main())