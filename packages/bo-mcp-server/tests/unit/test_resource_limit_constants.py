"""Resource-view limits must import their tool-view counterparts, not copy them.

``campaign_resource.py`` renders the same campaign data as
``bo_list_campaigns``; if its cap were a separately-declared literal, a
change to the tool's ``MAX_LIMIT`` would silently stop applying to the
resource view. This test pins the import (identity, not just equal
value) so that drift is impossible by construction.
"""

from __future__ import annotations

from bo_mcp_server.operations.list_campaigns import MAX_LIMIT
from bo_mcp_server.resources import campaign_resource


def test_campaign_resource_limit_is_the_same_object_as_operation_max_limit() -> None:
    assert campaign_resource.MAX_LIMIT is MAX_LIMIT
