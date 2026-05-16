"""Temporary development authentication helpers.

This module exists only to support local development while API-key validation
is bypassed. The actual user-bootstrap logic lives in
:mod:`bo_mcp_server.client` so this transport layer does not import from
storage directly; this file re-exports the helper for backwards
compatibility with existing imports.
"""

from bo_mcp_server.client import ensure_dev_user

__all__ = ["ensure_dev_user"]
