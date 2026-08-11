"""Drift guard: TOOL_SCHEMAS.md input schemas match the registered tools.

Scope: the result-ingestion pair (``bo_submit_results``,
``bo_upload_results_file``), whose documented input schemas drive agent
retry and duplicate-recovery behavior — a stale entry there makes agents
omit operational controls (idempotency, atomicity) or attempt unsupported
recovery (``force`` on upload). The remaining tool entries are not yet
guarded; extending coverage requires first reconciling their documented
blocks with the runtime schemas.

Mechanism: the fenced JSON block under each guarded tool's
``**Input Schema:**`` heading is parsed and its top-level keys are
compared for equality against the tool's runtime ``inputSchema``
properties, so adding, renaming, or removing a parameter without updating
the document fails here — in either direction.
"""

import json
import re
from pathlib import Path

import pytest

from bo_mcp_server.server import create_mcp_server

_TOOL_SCHEMAS_DOC = Path(__file__).resolve().parents[2] / "TOOL_SCHEMAS.md"

_GUARDED_TOOLS = ("bo_submit_results", "bo_upload_results_file")


def _documented_input_keys(doc_text: str, tool_name: str) -> set[str]:
    """Top-level keys of the tool's documented ``**Input Schema:**`` block."""
    section_match = re.search(
        rf"### `{tool_name}`\n(.*?)(?=\n## |\n### |\Z)",
        doc_text,
        re.DOTALL,
    )
    assert section_match, f"no section for {tool_name} in {_TOOL_SCHEMAS_DOC.name}"
    block_match = re.search(
        r"\*\*Input Schema:\*\*\s*```json\n(.*?)```",
        section_match.group(1),
        re.DOTALL,
    )
    assert block_match, f"no documented input schema for {tool_name}"
    return set(json.loads(block_match.group(1)))


@pytest.fixture(scope="module")
def runtime_input_properties() -> dict[str, set[str]]:
    server = create_mcp_server()
    return {
        tool.name: set(tool.parameters.get("properties", {}))
        for tool in server._tool_manager.list_tools()
    }


@pytest.mark.parametrize("tool_name", _GUARDED_TOOLS)
def test_documented_input_schema_matches_runtime(
    tool_name: str, runtime_input_properties: dict[str, set[str]]
) -> None:
    doc_text = _TOOL_SCHEMAS_DOC.read_text(encoding="utf-8")
    documented = _documented_input_keys(doc_text, tool_name)
    actual = runtime_input_properties[tool_name]
    assert documented == actual, (
        f"{tool_name}: TOOL_SCHEMAS.md input schema drifted from the "
        f"registered tool — undocumented: {sorted(actual - documented)}, "
        f"documented but not accepted: {sorted(documented - actual)}"
    )
