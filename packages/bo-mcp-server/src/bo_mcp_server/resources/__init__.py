"""MCP Resources for Bayesian Optimization.

Tenant-scoping model (TODO 8.40 friend-review finding)
------------------------------------------------------

MCP resources here read the full active dataset — they do **not** scope
to a per-caller owner. The friend-review for the May 2026 audit
flagged this for the new ``campaigns://recent`` surface and the
fuzzy-match suggestions attached to ``CAMPAIGN_NOT_FOUND``, but the
behaviour was already present on every existing resource
(``campaigns://list``, ``campaign://{id}``, ``suggestions://{id}``,
``suggestion://{id}``, ``events://{id}``) — none of them thread caller
identity into the storage layer.

This is intentional under the codebase's current deployment model:

1. **Per-process single-tenant.** The MCP transport is configured for
   ``127.0.0.1`` / ``localhost`` / ``mcp:*`` by default (see
   :func:`bo_mcp_server.server._build_allowed_hosts`). Production
   deployments run one MCP server per user; multi-tenant access goes
   through the REST surface, which is authenticated via
   ``CurrentUser`` and scopes every query at the route layer.
2. **No identity on the resource handler.** FastMCP resource handlers
   receive no session / user context, only the parsed URI parameters.
   Re-threading identity would require either a global ``ContextVar``
   set by the transport or a server-instance-per-user model.

If a future deployment serves multiple tenants from a single MCP
process, the discovery surfaces here must be scoped: the
``CampaignRepository.list_recent`` / ``list_active_ids`` methods
already accept an ``owner_id`` keyword, and the fuzzy-match helper
takes the candidate pool as input — both can be tightened without
changing their public shape. Until that need exists, the resource
layer remains process-scoped and the REST API is the multi-tenant
boundary.
"""

from bo_mcp_server.resources import campaign_resource, events_resource, suggestion_resource

__all__ = ["campaign_resource", "events_resource", "suggestion_resource"]
