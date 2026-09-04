"""Tests for the BO specialist's custom deep-agent factory."""

import os
from unittest.mock import patch

from prompts import BO_SPECIALIST_INSTRUCTIONS
from pydantic_ai.models.test import TestModel
from specialist import build_bo_specialist_agent, build_bo_specialist_subagent


def test_specialist_does_not_penalize_isolated_failures() -> None:
    assert "Do not assign strong penalties to failed experiments" in BO_SPECIALIST_INSTRUCTIONS
    assert "isolated failures may be random" in BO_SPECIALIST_INSTRUCTIONS


def test_specialist_factory_does_not_prepend_default_instructions() -> None:
    """The specialist receives only its domain instructions as the base prompt."""
    model = TestModel()
    with patch.dict(os.environ, {}, clear=True):
        config = build_bo_specialist_subagent(model)

    with patch("specialist.create_deep_agent") as create_deep_agent:
        build_bo_specialist_agent(config)

    kwargs = create_deep_agent.call_args.kwargs
    assert kwargs["model"] is model
    assert kwargs["instructions"] == BO_SPECIALIST_INSTRUCTIONS
    assert kwargs["include_filesystem"] is True
    assert kwargs["include_execute"] is True
    assert kwargs["include_todo"] is True
    assert kwargs["extra_toolsets"] == tuple(config["toolsets"])
