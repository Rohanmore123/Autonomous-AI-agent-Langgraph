"""
app/middleware/error_handler.py
================================
Global exception handler middleware.

WHY CENTRALISED ERROR HANDLING?
  Without this, unhandled exceptions bubble up as 500 Internal Server Error
  with raw Python tracebacks — exposing stack frames to clients (security risk)
  and returning inconsistent response shapes.

  This middleware:
    • Catches ALL unhandled exceptions
    • Maps known exception types to appropriate HTTP status codes
    • Returns the same ErrorResponse JSON shape regardless of error type
    • Logs the full traceback internally (never sent to client)
    • Includes request_id in response so the client can reference it in support tickets

EXCEPTION MAP:
  ValueError            → 400 Bad Request
  PermissionError       → 403 Forbidden
  FileNotFoundError     → 404 Not Found
  TimeoutError          → 504 Gateway Timeout
  RuntimeError          → 503 Service Unavailable  (used for LLM failures)
  HTTPException         → passthrough (handled by FastAPI)
  Everything else       → 500 Internal Server Error

SECURITY PRINCIPLE:
  Never expose internal error details (file paths, SQL queries, stack frames)
  in API responses. Log them server-side only.
"""

from __future__ import annotations

import traceback

from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.core.logging import get_logger

logger = get_logger(__name__)

# Maps exception type → (http_status, error_code)
EXCEPTION_MAP: dict[type[Exception], tuple[int, str]] = {
    ValueError:          (400, "INVALID_INPUT"),
    PermissionError:     (403, "PERMISSION_DENIED"),
    FileNotFoundError:   (404, "NOT_FOUND"),
    TimeoutError:        (504, "GATEWAY_TIMEOUT"),
    RuntimeError:        (503, "SERVICE_UNAVAILABLE"),
    NotImplementedError: (501, "NOT_IMPLEMENTED"),
    ConnectionError:     (503, "SERVICE_UNAVAILABLE"),
}


def _error_response(
    status_code: int,
    code: str,
    message: str,
    request_id: str | None = None,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "errors": [{"code": code, "message": message}],
            "request_id": request_id,
        },
    )


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = getattr(request.state, "request_id", None)

        try:
            return await call_next(request)

        except HTTPException as exc:
            # FastAPI HTTP exceptions — re-raise so FastAPI handles them normally
            # but wrap in our response envelope
            return _error_response(
                status_code=exc.status_code,
                code="HTTP_ERROR",
                message=str(exc.detail),
                request_id=request_id,
            )

        except Exception as exc:
            # Log full traceback internally
            logger.error(
                "unhandled_exception",
                exc_type=type(exc).__name__,
                exc_message=str(exc),
                traceback=traceback.format_exc(),
            )

            # Map to HTTP status
            for exc_type, (status_code, code) in EXCEPTION_MAP.items():
                if isinstance(exc, exc_type):
                    return _error_response(
                        status_code=status_code,
                        code=code,
                        # Safe message — no internal details
                        message=f"{exc_type.__name__}: {str(exc)[:200]}",
                        request_id=request_id,
                    )

            # Fallback: 500
            return _error_response(
                status_code=500,
                code="INTERNAL_SERVER_ERROR",
                message="An unexpected error occurred. Please try again or contact support.",
                request_id=request_id,
            )


# ---------------------------------------------------------------------------
# FastAPI exception handlers (registered on the app object, not middleware)
# ---------------------------------------------------------------------------

async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """
    Handle Pydantic validation errors (422 Unprocessable Entity).
    Returns field-level error details to help clients fix bad requests.
    """
    request_id = getattr(request.state, "request_id", None)
    errors = [
        {
            "code":    "VALIDATION_ERROR",
            "message": " → ".join(str(loc) for loc in err["loc"]) + ": " + err["msg"],
            "field":   ".".join(str(loc) for loc in err["loc"]),
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={"success": False, "errors": errors, "request_id": request_id},
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Wrap FastAPI HTTPExceptions in our standard envelope."""
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "errors": [{"code": "HTTP_ERROR", "message": str(exc.detail)}],
            "request_id": request_id,
        },
        headers=getattr(exc, "headers", None),
    )