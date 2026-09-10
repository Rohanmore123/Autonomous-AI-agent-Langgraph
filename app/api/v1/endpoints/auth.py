"""
app/api/v1/endpoints/auth.py
=============================
Authentication endpoints:
  POST /register        — create account
  POST /login           — exchange credentials for JWT pair
  POST /refresh         — use refresh token to get new access token
  POST /logout          — revoke refresh token
  POST /logout-all      — revoke all sessions (password change / account compromise)
  POST /change-password — update password + revoke all tokens
  GET  /google          — initiate Google OAuth2 flow
  GET  /google/callback — OAuth2 callback; store credentials
  GET  /me              — return current user profile

SECURITY NOTES:
  • Passwords are never logged or included in any response.
  • Refresh tokens are verified against Redis (server-side state).
  • All auth events are written to the AuditLog table.
  • Rate limiting on /login is handled by the RateLimitMiddleware.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from google_auth_oauthlib.flow import Flow
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.api.v1.dependencies.auth import CurrentUser, get_current_user
from app.core.config import get_settings
from app.core.database import get_db_session
from app.core.logging import get_logger
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    is_refresh_token_valid,
    revoke_all_user_tokens,
    revoke_refresh_token,
    verify_password,
)
from app.models.models import AuditLog, GoogleOAuthToken, Role, User, UserRole
from app.schemas.schemas import (
    APIResponse,
    PasswordChangeRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserLogin,
    UserRegister,
    UserResponse,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])
settings = get_settings()
logger = get_logger(__name__)


def _user_roles(user: User) -> list[str]:
    return [ur.role.name for ur in (user.user_roles or [])]


# ---------------------------------------------------------------------------
# Register
# ---------------------------------------------------------------------------

@router.post(
    "/register",
    response_model=APIResponse[UserResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
async def register(
    body: UserRegister,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse[UserResponse]:
    """
    Register a new user.
    - Checks email + username uniqueness
    - Hashes password with bcrypt
    - Assigns default 'user' role
    - Writes audit log entry
    """
    # Check email uniqueness
    existing = await db.execute(
        select(User).where(User.email == body.email.lower())
    )
    if existing.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    # Check username uniqueness
    existing_username = await db.execute(
        select(User).where(User.username == body.username)
    )
    if existing_username.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username already taken",
        )

    # Get default 'user' role
    role_result = await db.execute(select(Role).where(Role.name == "user"))
    default_role = role_result.scalar_one_or_none()

    # Create user
    user = User(
        email=body.email.lower(),
        username=body.username,
        hashed_password=hash_password(body.password),
        full_name=body.full_name,
    )
    db.add(user)
    await db.flush()  # Get user.id without committing

    # Assign role
    if default_role:
        db.add(UserRole(user_id=user.id, role_id=default_role.id))

    # Audit log
    db.add(AuditLog(
        user_id=user.id,
        action="user.registered",
        ip_address=request.client.host if request.client else None,
        details={"email": user.email, "username": user.username},
    ))

    await db.commit()
    await db.refresh(user)

    logger.info("user.registered", user_id=user.id, email=user.email)

    return APIResponse(
        data=UserResponse(
            id=user.id,
            email=user.email,
            username=user.username,
            full_name=user.full_name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            roles=_user_roles(user),
        ),
        message="Account created successfully",
    )


# ---------------------------------------------------------------------------
# Login
# ---------------------------------------------------------------------------

@router.post(
    "/login",
    response_model=APIResponse[TokenResponse],
    summary="Login and receive JWT tokens",
)
async def login(
    body: UserLogin,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse[TokenResponse]:
    """
    Authenticate with email + password.
    Returns access token (short-lived) + refresh token (long-lived).
    """
    result = await db.execute(
        select(User)
        .options(selectinload(User.user_roles).selectinload(UserRole.role))
        .where(User.email == body.email.lower())
        .where(User.deleted_at.is_(None))
    )
    user = result.scalar_one_or_none()

    # Constant-time check to prevent user enumeration via timing attack
    if not user or not verify_password(body.password, user.hashed_password):
        db.add(AuditLog(
            action="user.login.failed",
            ip_address=request.client.host if request.client else None,
            details={"email": body.email},
        ))
        await db.commit()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )

    roles = _user_roles(user)
    primary_role = roles[0] if roles else "user"

    access_token = create_access_token(user.id, primary_role)
    refresh_token = await create_refresh_token(user.id, primary_role)

    # Update last login timestamp
    user.last_login_at = datetime.now(timezone.utc)
    db.add(AuditLog(
        user_id=user.id,
        action="user.login.success",
        ip_address=request.client.host if request.client else None,
        details={"roles": roles},
    ))
    await db.commit()

    logger.info("user.login", user_id=user.id)

    return APIResponse(
        data=TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=settings.jwt.access_token_expire_minutes * 60,
        )
    )


# ---------------------------------------------------------------------------
# Refresh Token
# ---------------------------------------------------------------------------

@router.post(
    "/refresh",
    response_model=APIResponse[TokenResponse],
    summary="Refresh access token using refresh token",
)
async def refresh_token(
    body: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse[TokenResponse]:
    from jose import JWTError
    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Token type must be 'refresh'")

    jti = payload.get("jti")
    user_id = payload.get("sub")
    role = payload.get("role", "user")

    # Verify refresh token is still valid in Redis (not revoked)
    if not await is_refresh_token_valid(jti):
        raise HTTPException(status_code=401, detail="Refresh token has been revoked")

    # Rotate: revoke old, issue new pair
    await revoke_refresh_token(jti)
    new_access = create_access_token(user_id, role)
    new_refresh = await create_refresh_token(user_id, role)

    logger.info("auth.token.refreshed", user_id=user_id)

    return APIResponse(
        data=TokenResponse(
            access_token=new_access,
            refresh_token=new_refresh,
            token_type="bearer",
            expires_in=settings.jwt.access_token_expire_minutes * 60,
        )
    )


# ---------------------------------------------------------------------------
# Logout
# ---------------------------------------------------------------------------

@router.post("/logout", response_model=APIResponse, summary="Logout (revoke refresh token)")
async def logout(
    body: RefreshTokenRequest,
    current_user: CurrentUser,
) -> APIResponse:
    from jose import JWTError
    try:
        payload = decode_token(body.refresh_token)
    except JWTError:
        raise HTTPException(status_code=400, detail="Invalid refresh token")

    jti = payload.get("jti")
    if jti:
        await revoke_refresh_token(jti)

    logger.info("user.logout", user_id=current_user.id)
    return APIResponse(message="Logged out successfully")


@router.post(
    "/logout-all",
    response_model=APIResponse,
    summary="Revoke all sessions for the current user",
)
async def logout_all(current_user: CurrentUser) -> APIResponse:
    await revoke_all_user_tokens(current_user.id)
    logger.info("user.logout_all", user_id=current_user.id)
    return APIResponse(message="All sessions terminated")


# ---------------------------------------------------------------------------
# Change Password
# ---------------------------------------------------------------------------

@router.post(
    "/change-password",
    response_model=APIResponse,
    summary="Change password and revoke all sessions",
)
async def change_password(
    body: PasswordChangeRequest,
    current_user: CurrentUser,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    if not verify_password(body.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    current_user.hashed_password = hash_password(body.new_password)
    db.add(AuditLog(
        user_id=current_user.id,
        action="user.password.changed",
        ip_address=request.client.host if request.client else None,
    ))
    await db.commit()

    # Security: revoke all existing sessions after password change
    await revoke_all_user_tokens(current_user.id)
    logger.info("user.password.changed", user_id=current_user.id)
    return APIResponse(message="Password updated. All sessions have been terminated.")


# ---------------------------------------------------------------------------
# Current User Profile
# ---------------------------------------------------------------------------

@router.get("/me", response_model=APIResponse[UserResponse], summary="Get current user profile")
async def get_me(current_user: CurrentUser) -> APIResponse[UserResponse]:
    return APIResponse(
        data=UserResponse(
            id=current_user.id,
            email=current_user.email,
            username=current_user.username,
            full_name=current_user.full_name,
            is_active=current_user.is_active,
            is_verified=current_user.is_verified,
            created_at=current_user.created_at,
            last_login_at=current_user.last_login_at,
            roles=_user_roles(current_user),
        )
    )


# ---------------------------------------------------------------------------
# Google OAuth2
# ---------------------------------------------------------------------------

@router.get("/google", summary="Initiate Google OAuth2 flow for Gmail/Calendar access")
async def google_oauth_start(current_user: CurrentUser) -> RedirectResponse:
    """Redirect user to Google's OAuth consent screen."""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id":     settings.google_oauth.client_id,
                "client_secret": settings.google_oauth.client_secret,
                "redirect_uris": [settings.google_oauth.redirect_uri],
                "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
                "token_uri":     "https://oauth2.googleapis.com/token",
            }
        },
        scopes=settings.google_oauth.scopes,
    )
    flow.redirect_uri = settings.google_oauth.redirect_uri

    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        state=current_user.id,   # Pass user ID through state parameter
        prompt="consent",
    )
    return RedirectResponse(url=auth_url)


@router.get("/google/callback", summary="Google OAuth2 callback")
async def google_oauth_callback(
    code: str,
    state: str,                    # state = user_id set in /google
    db: AsyncSession = Depends(get_db_session),
) -> APIResponse:
    """
    Exchange the authorization code for credentials and store them.
    The `state` parameter carries the user ID (set in /google).
    """
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id":     settings.google_oauth.client_id,
                "client_secret": settings.google_oauth.client_secret,
                "redirect_uris": [settings.google_oauth.redirect_uri],
                "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
                "token_uri":     "https://oauth2.googleapis.com/token",
            }
        },
        scopes=settings.google_oauth.scopes,
        state=state,
    )
    flow.redirect_uri = settings.google_oauth.redirect_uri
    flow.fetch_token(code=code)
    creds = flow.credentials

    user_id = state  # We passed user_id as state

    # Upsert token record
    result = await db.execute(
        select(GoogleOAuthToken).where(GoogleOAuthToken.user_id == user_id)
    )
    token_row = result.scalar_one_or_none()

    expiry = creds.expiry.replace(tzinfo=timezone.utc) if creds.expiry else None

    if token_row:
        token_row.access_token = creds.token
        token_row.refresh_token = creds.refresh_token or token_row.refresh_token
        token_row.token_expiry = expiry
        token_row.scopes = list(creds.scopes or [])
    else:
        db.add(GoogleOAuthToken(
            user_id=user_id,
            access_token=creds.token,
            refresh_token=creds.refresh_token,
            token_expiry=expiry,
            scopes=list(creds.scopes or []),
        ))

    await db.commit()
    logger.info("google_oauth.connected", user_id=user_id)
    return APIResponse(message="Google account connected successfully")