#!/bin/bash
uv sync --extra dev
uv run --no-sync pytest -p no:warnings --ignore=tests/integration/test_api_endpoints.py
