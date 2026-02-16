#!/bin/bash
uv run pytest -p no:warnings --ignore=tests/integration/test_api_endpoints.py
