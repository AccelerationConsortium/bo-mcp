"""Structured error system for MCP tools.

Provides error codes with recovery actions to help agents handle failures.
Each error includes a machine-readable code, human-readable message, and
actionable recovery guidance.

Usage:
    from bo_mcp_server.errors import ErrorCode, make_error_response

    # Simple error:
    return make_error_response(ErrorCode.CAMPAIGN_NOT_FOUND)

    # With custom message and details:
    return make_error_response(
        ErrorCode.CAMPAIGN_NOT_FOUND,
        message=f"Campaign {campaign_id} not found",
        details={"campaign_id": campaign_id},
    )
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Error codes for MCP tool failures.

    Codes are organized by category:
    - E0xx: Validation errors (input, state, format issues)
    - E1xx: Processing errors (model fitting, database, computation)
    """

    # Validation errors (E0xx)
    INVALID_CAMPAIGN_ID = "E001"
    CAMPAIGN_NOT_FOUND = "E002"
    INVALID_STATE_TRANSITION = "E003"
    DUPLICATE_RESULT = "E004"
    VALIDATION_FAILED = "E005"
    MISSING_PARAMETERS = "E006"
    MISSING_OBJECTIVES = "E007"
    CONSTRAINT_VIOLATION = "E008"
    SUGGESTION_NOT_FOUND = "E009"

    # Processing errors (E1xx)
    MODEL_FITTING_FAILED = "E101"
    ACQUISITION_OPTIMIZATION_FAILED = "E102"
    DATABASE_ERROR = "E103"
    INSUFFICIENT_DATA = "E104"


@dataclass
class StructuredError:
    """Structured error with recovery guidance.

    Attributes:
        code: Error code from ErrorCode enum.
        message: Human-readable error description.
        recovery_action: Actionable guidance for the agent.
        details: Optional additional context about the error.
    """

    code: ErrorCode
    message: str
    recovery_action: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON response.

        Returns:
            Dictionary with code, message, recovery_action, and optionally details.
        """
        result: dict[str, Any] = {
            "code": self.code.value,
            "message": self.message,
            "recovery_action": self.recovery_action,
        }
        if self.details:
            result["details"] = self.details
        return result


# Error recovery action catalog
# Maps each error code to a specific, actionable recovery instruction
ERROR_RECOVERY: dict[ErrorCode, str] = {
    ErrorCode.INVALID_CAMPAIGN_ID: (
        "Verify campaign_id is a valid UUID v4. Use campaigns://list resource to get valid IDs."
    ),
    ErrorCode.CAMPAIGN_NOT_FOUND: (
        "Use campaigns://list resource to verify campaign exists and get correct ID."
    ),
    ErrorCode.INVALID_STATE_TRANSITION: (
        "Check current status with campaign://{id} resource. "
        "Valid transitions: CREATED->RUNNING, RUNNING<->PAUSED, *->COMPLETED"
    ),
    ErrorCode.DUPLICATE_RESULT: (
        "Use force=True parameter to override duplicate detection, or skip this result."
    ),
    ErrorCode.VALIDATION_FAILED: (
        "Review the errors array, fix the issues, and retry validate_intake."
    ),
    ErrorCode.MISSING_PARAMETERS: (
        "Add at least one parameter to the intake_data.parameters array."
    ),
    ErrorCode.MISSING_OBJECTIVES: (
        "Add at least one objective to the intake_data.objectives array."
    ),
    ErrorCode.CONSTRAINT_VIOLATION: (
        "Check that constraint parameters exist and bounds are valid."
    ),
    ErrorCode.SUGGESTION_NOT_FOUND: (
        "Verify suggestion_id is correct. Use suggestions://{campaign_id} to list suggestions."
    ),
    ErrorCode.MODEL_FITTING_FAILED: (
        "Check data quality with get_diagnostics. May need more observations (minimum 2)."
    ),
    ErrorCode.ACQUISITION_OPTIMIZATION_FAILED: (
        "Try reducing batch_size or check for constraint conflicts."
    ),
    ErrorCode.DATABASE_ERROR: (
        "Retry the operation. If persistent, check DATABASE_URL configuration."
    ),
    ErrorCode.INSUFFICIENT_DATA: (
        "Submit more results before generating suggestions. "
        "Need at least 2 observations for model fitting."
    ),
}


# Default error messages for each error code
DEFAULT_MESSAGES: dict[ErrorCode, str] = {
    ErrorCode.INVALID_CAMPAIGN_ID: "Invalid campaign_id format",
    ErrorCode.CAMPAIGN_NOT_FOUND: "Campaign not found",
    ErrorCode.INVALID_STATE_TRANSITION: "Invalid campaign state for this operation",
    ErrorCode.DUPLICATE_RESULT: "Duplicate result detected",
    ErrorCode.VALIDATION_FAILED: "Intake validation failed",
    ErrorCode.MISSING_PARAMETERS: "At least one parameter is required",
    ErrorCode.MISSING_OBJECTIVES: "At least one objective is required",
    ErrorCode.CONSTRAINT_VIOLATION: "Constraint validation failed",
    ErrorCode.SUGGESTION_NOT_FOUND: "Suggestion not found",
    ErrorCode.MODEL_FITTING_FAILED: "Model fitting failed",
    ErrorCode.ACQUISITION_OPTIMIZATION_FAILED: "Acquisition optimization failed",
    ErrorCode.DATABASE_ERROR: "Database operation failed",
    ErrorCode.INSUFFICIENT_DATA: "Insufficient data for operation",
}


def make_error_response(
    code: ErrorCode,
    message: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a standardized error response with recovery guidance.

    Args:
        code: Error code from ErrorCode enum.
        message: Optional custom message (uses default if not provided).
        details: Optional additional details about the error.

    Returns:
        Dictionary with:
            - success: False
            - error: Structured error dict with code, message, recovery_action
            - errors: List with the error message (for backward compatibility)

    Example:
        >>> make_error_response(ErrorCode.CAMPAIGN_NOT_FOUND)
        {
            "success": False,
            "error": {
                "code": "E002",
                "message": "Campaign not found",
                "recovery_action": "Use campaigns://list resource..."
            },
            "errors": ["Campaign not found"]
        }

        >>> make_error_response(
        ...     ErrorCode.CAMPAIGN_NOT_FOUND,
        ...     message="Campaign abc-123 not found",
        ...     details={"campaign_id": "abc-123"}
        ... )
        {
            "success": False,
            "error": {
                "code": "E002",
                "message": "Campaign abc-123 not found",
                "recovery_action": "Use campaigns://list resource...",
                "details": {"campaign_id": "abc-123"}
            },
            "errors": ["Campaign abc-123 not found"]
        }
    """
    error_message = message or DEFAULT_MESSAGES.get(code, "Unknown error")
    recovery_action = ERROR_RECOVERY.get(code, "Contact support.")

    error = StructuredError(
        code=code,
        message=error_message,
        recovery_action=recovery_action,
        details=details,
    )

    return {
        "success": False,
        "error": error.to_dict(),
        "errors": [error_message],  # Backward compatibility
    }
