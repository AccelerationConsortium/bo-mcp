#!/bin/sh

set -eu

uv pip install \
  --python /opt/venv/bin/python \
  --no-deps \
  --no-build-isolation \
  -e /workspace/packages/bo-engine \
  -e /workspace/packages/bo-engine-baybe \
  -e /workspace/packages/bo-mcp-server \
  -e /workspace/packages/bo-mcp-api

exec "$@"
