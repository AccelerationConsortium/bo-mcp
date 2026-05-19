"""Unit tests for the field-error path helpers.

The dotted-path grammar matches conventional Python attribute access so
agents can copy the path directly back into a payload edit:
``parameters[0].name`` reads as ``payload['parameters'][0]['name']``,
matching how popular validation toolchains surface paths (Pydantic
``loc``, JSONSchema ``instancePath``, RFC 6901 JSON Pointer rendered in
dotted form).

References:
    - JSON Schema validation paths
      (https://json-schema.org/draft/2020-12/json-schema-core)
      describe per-field error addressing as the canonical way for
      machines to consume validation failures.
    - Pydantic ``ValidationError.errors()`` returns a per-error ``loc``
      tuple (https://docs.pydantic.dev/latest/errors/usage_errors/) which
      the helpers here normalise into a dotted path.
"""

from pydantic import BaseModel, Field, ValidationError

from bo_mcp_server.field_errors import (
    add_row_field_error,
    field_error_messages,
    format_loc_path,
    merge_pydantic_row_errors,
    validation_errors_to_field_errors,
)


class TestFormatLocPath:
    """Path formatting matches the documented grammar."""

    def test_empty_loc_collapses_to_empty_string(self) -> None:
        assert format_loc_path(()) == ""

    def test_single_attr(self) -> None:
        assert format_loc_path(("name",)) == "name"

    def test_index_only_loc(self) -> None:
        assert format_loc_path((0,)) == "[0]"

    def test_attr_then_index_then_attr(self) -> None:
        assert format_loc_path(("parameters", 0, "name")) == "parameters[0].name"

    def test_nested_objective_path(self) -> None:
        # ``throughput`` is a regular identifier so the path uses dotted
        # access throughout. Python keywords like ``yield`` would force
        # the bracket-quote form (covered separately).
        assert (
            format_loc_path(("results", 5, "objective_values", "throughput"))
            == "results[5].objective_values.throughput"
        )

    def test_python_keyword_dict_key_is_bracket_quoted(self) -> None:
        assert (
            format_loc_path(("results", 5, "objective_values", "yield"))
            == "results[5].objective_values['yield']"
        )

    def test_non_identifier_dict_key_uses_bracket_quote(self) -> None:
        path = format_loc_path(("measurement_uncertainty", "noisy.value"))
        assert path == "measurement_uncertainty['noisy.value']"

    def test_dict_key_with_quote_is_escaped(self) -> None:
        assert format_loc_path(("a", "o'brien")) == "a['o\\'brien']"


class TestValidationErrorsToFieldErrors:
    """Pydantic ValidationError → field_errors conversion."""

    def _build_error(self) -> ValidationError:
        class _Inner(BaseModel):
            x: int = Field(..., gt=0)
            tag: str = Field(..., min_length=1)

        class _Outer(BaseModel):
            name: str = Field(..., min_length=1)
            items: list[_Inner]

        try:
            _Outer.model_validate(
                {
                    "name": "",
                    "items": [{"x": 0, "tag": ""}],
                }
            )
        except ValidationError as e:
            return e
        msg = "expected ValidationError"
        raise AssertionError(msg)

    def test_groups_by_field_path(self) -> None:
        error = self._build_error()
        field_errors = validation_errors_to_field_errors(error)
        # Each leaf path gets its own bucket; ``items[0].x`` and
        # ``items[0].tag`` must not collide.
        assert "name" in field_errors
        assert "items[0].x" in field_errors
        assert "items[0].tag" in field_errors
        assert all(isinstance(v, list) for v in field_errors.values())


class TestRowFieldError:
    """Per-row field error composition for submit_results."""

    def test_appends_under_results_prefix(self) -> None:
        bucket: dict[str, list[str]] = {}
        add_row_field_error(bucket, 5, "objective_values", "missing yield")
        assert bucket == {"results[5].objective_values": ["missing yield"]}

    def test_empty_field_path_targets_row(self) -> None:
        bucket: dict[str, list[str]] = {}
        add_row_field_error(bucket, 2, "", "row over budget")
        assert bucket == {"results[2]": ["row over budget"]}

    def test_bracketed_subpath_does_not_double_dot(self) -> None:
        bucket: dict[str, list[str]] = {}
        add_row_field_error(bucket, 1, "measurement_uncertainty['yield']", "negative std")
        assert "results[1].measurement_uncertainty['yield']" in bucket

    def test_multiple_errors_on_same_path_accumulate(self) -> None:
        bucket: dict[str, list[str]] = {}
        add_row_field_error(bucket, 0, "objective_values", "missing yield")
        add_row_field_error(bucket, 0, "objective_values", "missing throughput")
        assert bucket["results[0].objective_values"] == [
            "missing yield",
            "missing throughput",
        ]


class TestMergePydanticRowErrors:
    """Splicing per-row Pydantic errors under ``results[i]``."""

    def test_strips_leading_row_index_prefix(self) -> None:
        class _Row(BaseModel):
            x: int = Field(..., ge=0)

        try:
            _Row.model_validate({"x": -1})
        except ValidationError as e:
            bucket: dict[str, list[str]] = {}
            merge_pydantic_row_errors(bucket, 3, e)
            assert "results[3].x" in bucket


class TestFieldErrorMessages:
    """Flattening field_errors back to a list[str]."""

    def test_path_prefixed_messages(self) -> None:
        flat = list(
            field_error_messages(
                {
                    "results[0].x": ["bad", "very bad"],
                    "": ["body-level"],
                }
            )
        )
        assert "results[0].x: bad" in flat
        assert "results[0].x: very bad" in flat
        assert "body-level" in flat
