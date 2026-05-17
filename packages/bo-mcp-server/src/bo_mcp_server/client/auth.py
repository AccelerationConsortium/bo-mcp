"""Authorization helpers for transport layers.

Wraps repository-level lookups behind a small set of facade APIs so the
transport layers (FastAPI REST, MCP) do not have to reach into
:mod:`bo_mcp_server.storage` directly. Each helper raises one of the
client-facade exception types (:class:`NotFoundError` /
:class:`NotAuthorizedError`); transports translate those into their own
status codes / error envelopes.

The helpers intentionally return domain entities (``Campaign``,
``Suggestion``) — never ORM rows — so storage refactors stay internal.
"""

from __future__ import annotations

import hashlib
import logging
from uuid import UUID

from bo_mcp_server.domain import Campaign, CampaignSpec, Result, Suggestion, User
from bo_mcp_server.domain.suggestion import SuggestionStatus
from bo_mcp_server.storage import (
    CampaignRepository,
    CampaignSpecRepository,
    ResultRepository,
    SuggestionRepository,
    UserRepository,
    get_session,
)

logger = logging.getLogger(__name__)


class ClientError(Exception):
    """Base for client-facade errors."""


class NotFoundError(ClientError):
    """The requested entity does not exist."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(f"{resource} {identifier} not found")
        self.resource = resource
        self.identifier = identifier


class NotAuthorizedError(ClientError):
    """The caller is not authorized to access the requested entity."""

    def __init__(self, resource: str, identifier: str) -> None:
        super().__init__(f"Not authorized to access {resource} {identifier}")
        self.resource = resource
        self.identifier = identifier


class InvalidIdentifierError(ClientError):
    """The supplied string cannot be parsed as a UUID."""

    def __init__(self, name: str, value: str) -> None:
        super().__init__(f"Invalid {name} format: {value!r}")
        self.name = name
        self.value = value


def parse_uuid(value: str, name: str = "id") -> UUID:
    """Parse a UUID string, raising :class:`InvalidIdentifierError` on failure."""
    try:
        return UUID(value)
    except ValueError as exc:
        raise InvalidIdentifierError(name, value) from exc


async def authorize_campaign(campaign_id: str, user_id: UUID) -> Campaign:
    """Fetch a campaign and verify ``user_id`` owns it.

    Raises:
        InvalidIdentifierError: ``campaign_id`` is not a valid UUID.
        NotFoundError: No campaign with this id exists.
        NotAuthorizedError: ``user_id`` is not the campaign owner.
    """
    campaign_uuid = parse_uuid(campaign_id, "campaign_id")

    async with get_session() as session:
        repo = CampaignRepository(session)
        campaign = await repo.get(campaign_uuid)

    if campaign is None:
        raise NotFoundError("Campaign", campaign_id)
    if campaign.owner_id != user_id:
        raise NotAuthorizedError("campaign", campaign_id)
    return campaign


async def authorize_suggestion(suggestion_id: str, user_id: UUID) -> Suggestion:
    """Fetch a suggestion and verify ``user_id`` owns its parent campaign."""
    suggestion_uuid = parse_uuid(suggestion_id, "suggestion_id")

    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)
        campaign_repo = CampaignRepository(session)

        suggestion = await suggestion_repo.get(suggestion_uuid)
        if suggestion is None:
            raise NotFoundError("Suggestion", suggestion_id)

        campaign = await campaign_repo.get(suggestion.campaign_id)
        if campaign is None:
            raise NotFoundError("Campaign", str(suggestion.campaign_id))
        if campaign.owner_id != user_id:
            raise NotAuthorizedError("suggestion", suggestion_id)
    return suggestion


async def ensure_owned_campaigns(campaign_ids: list[str], user_id: UUID) -> None:
    """Reject the request if any ``campaign_ids`` belong to another owner.

    Invalid or missing IDs are intentionally ignored — operations surface
    those in their structured payloads. Only foreign-owned campaigns
    raise :class:`NotAuthorizedError` here.
    """
    async with get_session() as session:
        repo = CampaignRepository(session)
        for campaign_id in campaign_ids:
            try:
                campaign_uuid = UUID(campaign_id)
            except ValueError:
                continue
            campaign = await repo.get(campaign_uuid)
            if campaign is None:
                continue
            if campaign.owner_id != user_id:
                raise NotAuthorizedError("campaign", campaign_id)


async def get_campaign_with_spec(campaign_id: str, user_id: UUID) -> tuple[Campaign, CampaignSpec]:
    """Return the campaign and its spec, after owner authorization.

    **Convenience helper for legacy REST shape.** Backs the
    ``GET /api/campaigns/{id}`` and CSV-upload routes which need the
    full ``Campaign`` / ``CampaignSpec`` pair — there is no MCP
    operation that returns this exact tuple (operations either return
    paginated summaries or perform a side-effecting action). Owner
    authorization is enforced via :func:`authorize_campaign`.
    """
    campaign = await authorize_campaign(campaign_id, user_id)
    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        spec = await spec_repo.get(campaign.spec_id)
    if spec is None:
        raise NotFoundError("Campaign spec", str(campaign.spec_id))
    return campaign, spec


async def get_campaign_spec_by_id(spec_id: str) -> CampaignSpec:
    """Fetch a campaign spec by id (no ownership check — specs are immutable).

    .. warning::

        This helper exists only for internal call sites (e.g. operations
        that have already resolved ownership through the owning campaign).
        Transport-layer routes must use :func:`get_spec_for_user`
        instead — specs hold IP-sensitive parameter/objective/constraint
        shape and are not a tenant-global lookup key.
    """
    spec_uuid = parse_uuid(spec_id, "spec_id")
    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        spec = await spec_repo.get(spec_uuid)
    if spec is None:
        raise NotFoundError("Spec", spec_id)
    return spec


async def get_spec_for_user(spec_id: str, user_id: UUID) -> CampaignSpec:
    """Fetch a campaign spec only when ``user_id`` owns a campaign using it.

    Resolves ownership through the owning campaign instead of treating
    the spec UUID as a global key. Without this routing, anyone who knows
    or guesses another tenant's spec UUID receives the full parameter,
    objective, and constraint shape — likely IP-sensitive content for
    chemistry / manufacturing users.

    Raises ``NotFoundError`` both when the spec does not exist and when
    no campaign owned by ``user_id`` references it; the response is
    deliberately uniform so the route cannot leak spec existence to a
    foreign tenant.
    """
    spec_uuid = parse_uuid(spec_id, "spec_id")
    async with get_session() as session:
        spec_repo = CampaignSpecRepository(session)
        campaign_repo = CampaignRepository(session)
        spec = await spec_repo.get(spec_uuid)
        if spec is None:
            raise NotFoundError("Spec", spec_id)
        campaigns_for_owner, _ = await campaign_repo.list_filtered(owner_id=user_id)
    owns_campaign_with_spec = any(c.spec_id == spec_uuid for c in campaigns_for_owner)
    if not owns_campaign_with_spec:
        raise NotFoundError("Spec", spec_id)
    return spec


async def list_owner_campaigns_with_specs(user_id: UUID) -> list[tuple[Campaign, CampaignSpec]]:
    """List all campaigns owned by ``user_id`` paired with their resolved spec.

    **Convenience helper for legacy REST shape.** This intentionally does
    NOT route through :func:`list_campaigns_operation`: that operation
    returns paginated *summary dicts* keyed by ``verbosity``, whereas the
    historical ``GET /api/campaigns`` route must answer with full
    (Campaign, Spec) pairs as a bare array. Both views are first-class —
    MCP / ``POST /query`` callers use the operation; legacy GET callers
    use this helper. See ``TestConvenienceVsOperationDivergence`` for
    the pinned divergence.

    Uses a single ``get_by_ids`` batch to avoid an N+1 spec lookup;
    campaigns whose spec has been deleted are dropped (mirroring the
    historical REST behavior).
    """
    async with get_session() as session:
        campaign_repo = CampaignRepository(session)
        spec_repo = CampaignSpecRepository(session)

        campaigns = await campaign_repo.list_by_owner(user_id)
        if not campaigns:
            return []
        spec_ids = [c.spec_id for c in campaigns]
        specs_by_id = await spec_repo.get_by_ids(spec_ids)

    pairs: list[tuple[Campaign, CampaignSpec]] = []
    for campaign in campaigns:
        spec = specs_by_id.get(campaign.spec_id)
        if spec is not None:
            pairs.append((campaign, spec))
    return pairs


async def list_campaign_suggestions(
    campaign_id: str,
    user_id: UUID,
    status_filter: SuggestionStatus | None = None,
) -> list[Suggestion]:
    """List suggestions for a campaign after owner authorization.

    **Convenience helper for legacy REST shape.** The historical
    ``GET /api/suggestions/{campaign_id}`` route returns a bare array of
    full Suggestion entities, while the MCP / ``POST /query`` route uses
    :func:`list_suggestions_operation` and returns a paginated envelope
    of summary dicts. Both shapes are first-class; this helper exists so
    the legacy route does not have to reach into ``SuggestionRepository``
    itself. See ``TestConvenienceVsOperationDivergence`` for the pinned
    divergence.
    """
    campaign = await authorize_campaign(campaign_id, user_id)
    async with get_session() as session:
        suggestion_repo = SuggestionRepository(session)
        return await suggestion_repo.list_by_campaign(campaign.id, status_filter)


async def list_campaign_results(campaign_id: str, user_id: UUID) -> list[Result]:
    """List results for a campaign after owner authorization.

    **Convenience helper for legacy REST shape.** Mirrors
    :func:`list_campaign_suggestions`: the historical
    ``GET /api/results/{campaign_id}`` route returns a bare array of full
    Result entities, while MCP / ``POST /query`` callers consume the
    paginated envelope from :func:`list_results_operation`. Both shapes
    are first-class; see ``TestConvenienceVsOperationDivergence``.
    """
    campaign = await authorize_campaign(campaign_id, user_id)
    async with get_session() as session:
        result_repo = ResultRepository(session)
        return await result_repo.list_by_campaign(campaign.id)


# ---------------------------------------------------------------------------
# Development user (temporary auth bypass)
# ---------------------------------------------------------------------------


DEV_API_KEY = "dev-api-key-12345"
DEV_USER_NAME = "Test User"
DEV_USER_EMAIL = "test@example.com"


async def get_user_by_api_key(api_key: str) -> User | None:
    """Resolve an active user from a raw API key string.

    Hashes the key with SHA-256 — the same algorithm used to populate
    :attr:`UserModel.api_key_hash` — and returns the matching user only
    when their account is active. Deactivated users surface as ``None``
    so callers convert them into the same generic 401 as an unknown
    key, preventing an attacker (or a forgotten offboarding) from
    extending access after the account was disabled.

    Reference: storing API-key hashes (never plaintext) and refusing
    credentials for deactivated accounts is the standard pattern
    recommended by OWASP for service-to-service authentication — see
    https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html.
    """
    if not api_key:
        return None
    api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    async with get_session() as session:
        repo = UserRepository(session)
        user = await repo.get_by_api_key_hash(api_key_hash)
    if user is None or not user.is_active:
        return None
    return user


async def ensure_dev_user() -> User:
    """Ensure the shared development user exists, is active, and uses ``DEV_API_KEY``.

    The new real-auth path resolves callers by API-key hash and active
    status, so "a user named ``test@example.com`` exists" is no longer
    sufficient: a stale record left over from a previous schema (wrong
    hash, deactivated) would leave ``DEV_API_KEY`` unusable even though
    the bootstrap appeared to succeed. This helper therefore repairs
    those mismatches in place — local dev is the only context that ever
    calls it, so the heuristic is safe — and re-issues the canonical
    record otherwise.

    This is a temporary auth-bypass helper for local development; it
    lives in the facade so transports do not have to instantiate
    :class:`UserRepository` directly. Production deployments must
    replace the caller of this helper with a real authentication path.
    """
    api_key_hash = hashlib.sha256(DEV_API_KEY.encode()).hexdigest()
    async with get_session() as session:
        repo = UserRepository(session)
        existing = await repo.get_by_email(DEV_USER_EMAIL)
        if existing is not None:
            if existing.api_key_hash == api_key_hash and existing.is_active:
                return existing
            repaired = existing.model_copy(update={"api_key_hash": api_key_hash, "is_active": True})
            saved = await repo.save(repaired)
            logger.warning(
                "Repaired shared development user %s to match DEV_API_KEY and "
                "is_active=True. This must not happen in production.",
                saved.id,
            )
            return saved
        user = User(
            name=DEV_USER_NAME,
            email=DEV_USER_EMAIL,
            api_key_hash=api_key_hash,
        )
        saved = await repo.save(user)
    logger.warning(
        "Created shared development user %s for temporary auth bypass",
        saved.id,
    )
    return saved
