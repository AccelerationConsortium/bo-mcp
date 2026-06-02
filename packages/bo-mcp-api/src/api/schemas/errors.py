"""Reusable OpenAPI error response declarations."""

from typing import Any

from pydantic import BaseModel, Field

from api.schemas.common import API_RESPONSE_SCHEMA_VERSION


class HttpErrorResponse(BaseModel):
    """FastAPI HTTPException response body."""

    detail: str | dict[str, Any] | list[dict[str, Any]]


class ErrorInfo(BaseModel):
    """Structured operation/internal error details."""

    code: str
    message: str
    recovery_action: str | None = None
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class StructuredErrorEnvelope(BaseModel):
    """MCP-aligned structured error envelope."""

    schema_version: int = Field(default=API_RESPONSE_SCHEMA_VERSION)
    success: bool = False
    error: ErrorInfo


OpenApiResponses = dict[int | str, dict[str, Any]]


AUTH_ERROR_RESPONSES: OpenApiResponses = {
    401: {
        "model": HttpErrorResponse,
        "description": "Missing or invalid X-API-Key header.",
        "content": {
            "application/json": {
                "example": {"detail": "Authentication required"},
            },
        },
    },
}

COMMON_HTTP_ERROR_RESPONSES: OpenApiResponses = {
    **AUTH_ERROR_RESPONSES,
    400: {
        "model": HttpErrorResponse,
        "description": "Malformed identifier, invalid query combination, or invalid upload.",
    },
    403: {
        "model": HttpErrorResponse,
        "description": "Authenticated caller is not authorized to access this resource.",
    },
    404: {
        "model": HttpErrorResponse,
        "description": "Requested resource was not found.",
    },
    500: {
        "model": StructuredErrorEnvelope,
        "description": "Sanitized internal error envelope with request correlation details.",
    },
}

IDEMPOTENCY_ERROR_RESPONSES: OpenApiResponses = {
    409: {
        "model": HttpErrorResponse,
        "description": (
            "Idempotency conflict or in-progress operation. Reuse an "
            "Idempotency-Key only for retries of the exact same payload."
        ),
        "content": {
            "application/json": {
                "example": {
                    "detail": {
                        "code": "IDEMPOTENCY_CONFLICT",
                        "message": "Idempotency key was reused with a different payload.",
                        "retryable": False,
                    },
                },
            },
        },
    },
}


def operation_failure_response(
    *,
    description: str,
    example: dict[str, Any],
    model: type[BaseModel],
) -> dict[str, Any]:
    """Return OpenAPI docs for an operation-level ``success=false`` response."""
    return {
        "model": model,
        "description": description,
        "content": {"application/json": {"example": example}},
    }
