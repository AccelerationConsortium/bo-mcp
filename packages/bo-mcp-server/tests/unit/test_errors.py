"""Tests for the errors module.

These tests verify that the structured error system provides correct
error codes, messages, and recovery actions for agents.

References:
- Implementation Plan Section 12.2: Structured Error Codes with Recovery Actions
- MCP Specification: https://modelcontextprotocol.io/specification/2025-11-25
"""

from bo_mcp_server.errors import (
    DEFAULT_MESSAGES,
    ERROR_RECOVERY,
    ErrorCode,
    StructuredError,
    make_error_response,
)


class TestErrorCode:
    """Tests for ErrorCode enum."""

    def test_error_code_values_follow_convention(self) -> None:
        """Verify error codes follow E0xx (validation) and E1xx (processing) convention.

        This test ensures the error code numbering scheme is consistent.
        Reference: OpenAPI Error Codes best practice recommends structured codes.
        """
        # Validation errors should be E0xx
        validation_errors = [
            ErrorCode.INVALID_CAMPAIGN_ID,
            ErrorCode.CAMPAIGN_NOT_FOUND,
            ErrorCode.INVALID_STATE_TRANSITION,
            ErrorCode.DUPLICATE_RESULT,
            ErrorCode.VALIDATION_FAILED,
            ErrorCode.MISSING_PARAMETERS,
            ErrorCode.MISSING_OBJECTIVES,
            ErrorCode.CONSTRAINT_VIOLATION,
            ErrorCode.SUGGESTION_NOT_FOUND,
        ]
        for code in validation_errors:
            assert code.value.startswith("E0"), f"{code.name} should start with E0"

        # Processing errors should be E1xx
        processing_errors = [
            ErrorCode.MODEL_FITTING_FAILED,
            ErrorCode.ACQUISITION_OPTIMIZATION_FAILED,
            ErrorCode.DATABASE_ERROR,
            ErrorCode.INSUFFICIENT_DATA,
        ]
        for code in processing_errors:
            assert code.value.startswith("E1"), f"{code.name} should start with E1"

    def test_all_error_codes_have_unique_values(self) -> None:
        """Verify each error code has a unique value."""
        values = [code.value for code in ErrorCode]
        assert len(values) == len(set(values)), "Duplicate error code values found"

    def test_error_code_can_be_created_from_string(self) -> None:
        """Verify ErrorCode can be looked up by value."""
        # Should be able to find error code by value
        for code in ErrorCode:
            found = next((c for c in ErrorCode if c.value == code.value), None)
            assert found == code


class TestStructuredError:
    """Tests for StructuredError dataclass."""

    def test_to_dict_minimal(self) -> None:
        """Verify to_dict returns correct structure without details."""
        error = StructuredError(
            code=ErrorCode.CAMPAIGN_NOT_FOUND,
            message="Campaign abc-123 not found",
            recovery_action="Use campaigns://list resource",
        )

        result = error.to_dict()

        assert result["code"] == "E002"
        assert result["message"] == "Campaign abc-123 not found"
        assert result["recovery_action"] == "Use campaigns://list resource"
        assert "details" not in result

    def test_to_dict_with_details(self) -> None:
        """Verify to_dict includes details when provided."""
        error = StructuredError(
            code=ErrorCode.CAMPAIGN_NOT_FOUND,
            message="Campaign abc-123 not found",
            recovery_action="Use campaigns://list resource",
            details={"campaign_id": "abc-123", "searched_at": "2025-01-01"},
        )

        result = error.to_dict()

        assert "details" in result
        assert result["details"]["campaign_id"] == "abc-123"
        assert result["details"]["searched_at"] == "2025-01-01"


class TestErrorRecoveryMapping:
    """Tests for ERROR_RECOVERY mapping."""

    def test_all_error_codes_have_recovery_action(self) -> None:
        """Verify every error code has a recovery action.

        This is critical for agent usability - every error must have
        actionable guidance.
        """
        for code in ErrorCode:
            assert code in ERROR_RECOVERY, f"{code.name} missing recovery action"
            assert ERROR_RECOVERY[code], f"{code.name} has empty recovery action"

    def test_recovery_actions_are_actionable(self) -> None:
        """Verify recovery actions contain actionable instructions.

        Recovery actions should tell agents what to DO, not just what went wrong.
        """
        actionable_keywords = [
            "use",
            "verify",
            "check",
            "add",
            "retry",
            "review",
            "reduce",
            "submit",
        ]

        for code, recovery in ERROR_RECOVERY.items():
            recovery_lower = recovery.lower()
            has_action = any(keyword in recovery_lower for keyword in actionable_keywords)
            assert has_action, (
                f"{code.name} recovery action should contain actionable verb: {recovery}"
            )

    def test_recovery_actions_reference_resources_where_appropriate(self) -> None:
        """Verify recovery actions reference MCP resources where applicable."""
        # These error codes should reference specific MCP resources
        should_reference_resources = {
            ErrorCode.INVALID_CAMPAIGN_ID: "campaigns://list",
            ErrorCode.CAMPAIGN_NOT_FOUND: "campaigns://list",
            ErrorCode.INVALID_STATE_TRANSITION: "campaign://",
            ErrorCode.SUGGESTION_NOT_FOUND: "suggestions://",
        }

        for code, expected_resource in should_reference_resources.items():
            recovery = ERROR_RECOVERY[code]
            assert expected_resource in recovery, (
                f"{code.name} recovery should reference {expected_resource}"
            )


class TestDefaultMessages:
    """Tests for DEFAULT_MESSAGES mapping."""

    def test_all_error_codes_have_default_message(self) -> None:
        """Verify every error code has a default message."""
        for code in ErrorCode:
            assert code in DEFAULT_MESSAGES, f"{code.name} missing default message"
            assert DEFAULT_MESSAGES[code], f"{code.name} has empty default message"

    def test_default_messages_are_human_readable(self) -> None:
        """Verify default messages are clear and human-readable."""
        for code, message in DEFAULT_MESSAGES.items():
            # Should not be technical jargon or code
            assert not message.startswith("E0"), f"{code.name} message starts with error code"
            assert not message.startswith("Error:"), f"{code.name} redundant 'Error:' prefix"
            # Should be reasonable length
            assert len(message) < 100, f"{code.name} message too long: {len(message)} chars"


class TestMakeErrorResponse:
    """Tests for make_error_response function."""

    def test_basic_error_response(self) -> None:
        """Verify basic error response structure."""
        result = make_error_response(ErrorCode.CAMPAIGN_NOT_FOUND)

        assert result["success"] is False
        assert "error" in result
        assert "errors" in result

    def test_error_response_has_correct_structure(self) -> None:
        """Verify error response contains all required fields."""
        result = make_error_response(ErrorCode.CAMPAIGN_NOT_FOUND)

        error = result["error"]
        assert error["code"] == "E002"
        assert error["message"] == "Campaign not found"
        assert "recovery_action" in error
        assert len(error["recovery_action"]) > 0

    def test_custom_message_overrides_default(self) -> None:
        """Verify custom message replaces default."""
        result = make_error_response(
            ErrorCode.CAMPAIGN_NOT_FOUND,
            message="Campaign abc-123 not found in database",
        )

        assert result["error"]["message"] == "Campaign abc-123 not found in database"
        assert result["errors"] == ["Campaign abc-123 not found in database"]

    def test_details_are_included(self) -> None:
        """Verify details are passed through to response."""
        result = make_error_response(
            ErrorCode.CAMPAIGN_NOT_FOUND,
            details={"campaign_id": "abc-123", "searched_tables": ["campaigns"]},
        )

        assert "details" in result["error"]
        assert result["error"]["details"]["campaign_id"] == "abc-123"

    def test_backward_compatibility_errors_list(self) -> None:
        """Verify errors list is maintained for backward compatibility.

        The 'errors' list exists for compatibility with older tool responses
        that used simple string error lists.
        """
        result = make_error_response(ErrorCode.VALIDATION_FAILED)

        assert "errors" in result
        assert isinstance(result["errors"], list)
        assert len(result["errors"]) == 1
        assert result["errors"][0] == DEFAULT_MESSAGES[ErrorCode.VALIDATION_FAILED]

    def test_all_error_codes_produce_valid_response(self) -> None:
        """Verify all error codes can produce a valid response."""
        for code in ErrorCode:
            result = make_error_response(code)

            assert result["success"] is False
            assert result["error"]["code"] == code.value
            assert len(result["error"]["message"]) > 0
            assert len(result["error"]["recovery_action"]) > 0
            assert len(result["errors"]) == 1


class TestErrorResponseIntegration:
    """Integration tests for error response usage patterns."""

    def test_typical_campaign_not_found_response(self) -> None:
        """Test typical usage for campaign not found error.

        This represents how tools like get_diagnostics would use the error system.
        Reference: Pattern from generate_suggestions.py:315
        """
        campaign_id = "invalid-uuid-format"

        response = make_error_response(
            ErrorCode.CAMPAIGN_NOT_FOUND,
            message=f"Campaign {campaign_id} not found",
            details={"campaign_id": campaign_id},
        )

        # Verify agent can parse the response
        assert response["success"] is False
        assert response["error"]["code"] == "E002"
        assert campaign_id in response["error"]["message"]
        assert "campaigns://list" in response["error"]["recovery_action"]

    def test_typical_invalid_id_format_response(self) -> None:
        """Test typical usage for invalid UUID format error.

        This represents how tools validate campaign_id format.
        """
        response = make_error_response(ErrorCode.INVALID_CAMPAIGN_ID)

        # Verify response guides agent to fix the format
        assert response["success"] is False
        assert "UUID" in response["error"]["recovery_action"]
        assert "campaigns://list" in response["error"]["recovery_action"]

    def test_typical_duplicate_result_response(self) -> None:
        """Test typical usage for duplicate result detection.

        Reference: submit_results.py duplicate detection pattern
        """
        response = make_error_response(
            ErrorCode.DUPLICATE_RESULT,
            message="Result at index 2 is a duplicate of existing result",
            details={
                "result_index": 2,
                "duplicate_of_index": 5,
                "parameter_distance": 0.0,
            },
        )

        # Verify response tells agent how to proceed
        assert "force=True" in response["error"]["recovery_action"]
        assert response["error"]["details"]["result_index"] == 2

    def test_typical_model_fitting_failure(self) -> None:
        """Test typical usage for model fitting failure.

        This is a processing error that agents need to handle gracefully.
        """
        response = make_error_response(
            ErrorCode.MODEL_FITTING_FAILED,
            message="Model fitting failed: Singular matrix",
            details={"n_observations": 3, "n_parameters": 5},
        )

        # Verify response suggests checking data quality
        assert "get_diagnostics" in response["error"]["recovery_action"]
        assert "observations" in response["error"]["recovery_action"].lower()


class TestErrorCodeCoverage:
    """Tests ensuring error code coverage for common scenarios."""

    def test_validation_error_codes_cover_common_scenarios(self) -> None:
        """Verify validation error codes cover common input validation scenarios.

        Reference: Common MCP tool validation patterns
        """
        scenarios = {
            "invalid campaign_id format": ErrorCode.INVALID_CAMPAIGN_ID,
            "campaign does not exist": ErrorCode.CAMPAIGN_NOT_FOUND,
            "campaign in wrong state": ErrorCode.INVALID_STATE_TRANSITION,
            "duplicate data submission": ErrorCode.DUPLICATE_RESULT,
            "intake data invalid": ErrorCode.VALIDATION_FAILED,
            "no parameters provided": ErrorCode.MISSING_PARAMETERS,
            "no objectives provided": ErrorCode.MISSING_OBJECTIVES,
            "constraint configuration invalid": ErrorCode.CONSTRAINT_VIOLATION,
            "suggestion does not exist": ErrorCode.SUGGESTION_NOT_FOUND,
        }

        for scenario, expected_code in scenarios.items():
            assert expected_code in ErrorCode, f"Missing code for scenario: {scenario}"

    def test_processing_error_codes_cover_common_scenarios(self) -> None:
        """Verify processing error codes cover common runtime scenarios.

        Reference: Common BO computation failure modes
        """
        scenarios = {
            "GP model fitting failure": ErrorCode.MODEL_FITTING_FAILED,
            "acquisition function optimization failure": ErrorCode.ACQUISITION_OPTIMIZATION_FAILED,
            "database connection/query failure": ErrorCode.DATABASE_ERROR,
            "not enough data for operation": ErrorCode.INSUFFICIENT_DATA,
        }

        for scenario, expected_code in scenarios.items():
            assert expected_code in ErrorCode, f"Missing code for scenario: {scenario}"
