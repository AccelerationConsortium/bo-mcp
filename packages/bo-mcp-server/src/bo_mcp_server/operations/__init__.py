"""Transport-neutral operations shared by MCP tools and REST routes.

Operations contain all business logic. MCP tool handlers and HTTP route
handlers are thin wrappers that parse transport-specific arguments and
delegate to operations.
"""
