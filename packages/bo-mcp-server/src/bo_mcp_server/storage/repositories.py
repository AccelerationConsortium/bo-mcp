"""Repository implementations."""

import json
import logging
from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import (
    AcquisitionOptimizationConfig,
    Campaign,
    CampaignSpec,
    CampaignStatus,
    Constraint,
    ConstraintType,
    InputParameter,
    Objective,
    ParameterType,
    Result,
    Suggestion,
    SuggestionProvenance,
    SuggestionStatus,
    User,
)
from bo_mcp_server.domain.event import Event, EventType
from bo_mcp_server.domain.utils import utcnow
from bo_mcp_server.storage.base import ConcurrentModificationError
from bo_mcp_server.storage.models import (
    Base,
    CampaignModel,
    CampaignSpecModel,
    EventModel,
    ResultModel,
    SuggestionModel,
    UserModel,
)

logger = logging.getLogger(__name__)

# CampaignSpec attributes persisted via the ``advanced_options_json`` blob.
# They're consumed in-process (not queried in SQL), so a single JSON column
# is preferred over per-field columns — adding a new advanced field does
# not require another migration.
_ADVANCED_SPEC_FIELDS: tuple[str, ...] = (
    "acquisition_method",
    "use_input_warping",
    "use_cost_aware",
    "turbo_config",
    "saasbo_config",
    "fidelity_parameter",
    "transfer_learning",
    "outcome_constraints",
    "acknowledge_degradations",
)


def _serialize_advanced_options(spec: CampaignSpec) -> str | None:
    """Return a JSON blob with the advanced spec fields, or ``None`` when all default.

    Compared against the matching ``CampaignSpec()`` defaults so unchanged
    rows stay NULL — keeps the DB tidy and short-circuits the load path
    for the common case.
    """
    payload = spec.model_dump(mode="json", include=set(_ADVANCED_SPEC_FIELDS))
    defaults = CampaignSpec.model_construct().model_dump(
        mode="json", include=set(_ADVANCED_SPEC_FIELDS)
    )
    # ``model_construct`` skips validators; for our defaulted fields the values
    # match the field defaults, which is exactly what we want here.
    if payload == defaults:
        return None
    return json.dumps(payload)


def _advanced_options_kwargs(data: dict[str, Any]) -> dict[str, Any]:
    """Pick only known advanced fields from a deserialized JSON blob.

    Filters unknown keys (forward-compat with future advanced fields)
    and skips ``None``/missing entries so ``CampaignSpec`` defaults apply.
    """
    return {k: data[k] for k in _ADVANCED_SPEC_FIELDS if k in data and data[k] is not None}


def _active_filter(model: type[Base], include_deleted: bool) -> list[Any]:
    """Return WHERE-clauses that hide soft-deleted rows by default.

    TODO 8.11: every campaign / suggestion / result / event read goes
    through this helper so a single flag (``include_deleted=True``,
    reserved for admin / forensics paths) flips the filter
    consistently. The model's ``deleted_at`` column must exist for
    this helper to be valid; calling it on a model without the column
    silently disables the filter (and is therefore a bug — guard with
    an ``hasattr`` check before extending to a new entity).
    """
    if include_deleted:
        return []
    deleted_at = getattr(model, "deleted_at", None)
    if deleted_at is None:
        return []
    return [deleted_at.is_(None)]


class UserRepository:
    """Repository for User entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind this repository to the given async SQLAlchemy session."""
        self.session = session

    async def get(self, entity_id: UUID) -> User | None:
        """Get user by ID."""
        result = await self.session.execute(select(UserModel).where(UserModel.id == str(entity_id)))
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def get_by_email(self, email: str) -> User | None:
        """Get user by email."""
        result = await self.session.execute(select(UserModel).where(UserModel.email == email))
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def get_by_api_key_hash(self, api_key_hash: str) -> User | None:
        """Get user by API key hash."""
        result = await self.session.execute(
            select(UserModel).where(UserModel.api_key_hash == api_key_hash)
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def save(self, user: User) -> User:
        """Save user."""
        model = UserModel(
            id=str(user.id),
            name=user.name,
            email=user.email,
            api_key_hash=user.api_key_hash,
            is_active=user.is_active,
            created_at=user.created_at,
            last_active_at=user.last_active_at,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

    async def delete(self, entity_id: UUID) -> bool:
        """Delete user by ID."""
        result = await self.session.execute(select(UserModel).where(UserModel.id == str(entity_id)))
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    async def list_all(self) -> list[User]:
        """List all users."""
        result = await self.session.execute(select(UserModel))
        return [self._to_entity(m) for m in result.scalars()]

    def _to_entity(self, model: UserModel) -> User:
        """Convert ORM model to domain entity."""
        return User(
            id=UUID(model.id),
            name=model.name,
            email=model.email,
            api_key_hash=model.api_key_hash,
            is_active=model.is_active,
            created_at=model.created_at,
            last_active_at=model.last_active_at,
        )


class CampaignSpecRepository:
    """Repository for CampaignSpec entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind this repository to the given async SQLAlchemy session."""
        self.session = session

    async def get(self, entity_id: UUID) -> CampaignSpec | None:
        """Get campaign spec by ID."""
        result = await self.session.execute(
            select(CampaignSpecModel).where(CampaignSpecModel.id == str(entity_id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def save(self, spec: CampaignSpec, spec_id: UUID) -> CampaignSpec:
        """Save campaign spec with explicit ID (specs are immutable)."""
        acq_opt = spec.acquisition_optimization
        # ``backend_options`` is wrapped in nested ``MappingProxyType`` views
        # by the domain validator; ``json.dumps`` cannot serialize those
        # directly, so materialize a plain nested dict at the boundary.
        backend_options_json = (
            json.dumps({k: dict(v) for k, v in spec.backend_options.items()})
            if spec.backend_options
            else None
        )
        advanced_options_json = _serialize_advanced_options(spec)
        model = CampaignSpecModel(
            id=str(spec_id),
            name=spec.name,
            description=spec.description,
            parameters_json=json.dumps([p.model_dump() for p in spec.parameters]),
            objectives_json=json.dumps([o.model_dump() for o in spec.objectives]),
            constraints_json=json.dumps([c.model_dump() for c in spec.constraints]),
            batch_size=spec.batch_size,
            max_iterations=spec.max_iterations,
            max_observations=spec.max_observations,
            convergence_tolerance=spec.convergence_tolerance,
            acquisition_num_restarts=acq_opt.num_restarts if acq_opt is not None else None,
            acquisition_raw_samples=acq_opt.raw_samples if acq_opt is not None else None,
            initial_design_size=spec.initial_design_size,
            random_seed=spec.random_seed,
            backend=spec.backend,
            backend_options_json=backend_options_json,
            advanced_options_json=advanced_options_json,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

    async def delete(self, entity_id: UUID) -> bool:
        """Delete campaign spec by ID."""
        result = await self.session.execute(
            select(CampaignSpecModel).where(CampaignSpecModel.id == str(entity_id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    async def list_all(self) -> list[CampaignSpec]:
        """List all campaign specs."""
        result = await self.session.execute(select(CampaignSpecModel))
        return [self._to_entity(m) for m in result.scalars()]

    async def get_by_ids(self, ids: list[UUID]) -> dict[UUID, CampaignSpec]:
        """Get multiple campaign specs by their IDs in a single query.

        Args:
            ids: List of UUIDs to fetch

        Returns:
            Dictionary mapping UUID to CampaignSpec for found specs
        """
        if not ids:
            return {}

        str_ids = [str(value) for value in ids]
        result = await self.session.execute(
            select(CampaignSpecModel).where(CampaignSpecModel.id.in_(str_ids))
        )
        return {UUID(m.id): self._to_entity(m) for m in result.scalars()}

    def _to_entity(self, model: CampaignSpecModel) -> CampaignSpec:
        """Convert ORM model to domain entity."""
        parameters = [
            InputParameter(
                name=p["name"],
                type=ParameterType(p["type"]),
                bounds=p.get("bounds"),
                values=p.get("values"),
                categories=p.get("categories"),
                description=p.get("description", ""),
                parameter_options=p.get("parameter_options"),
            )
            for p in model.get_parameters()
        ]

        objectives = [
            Objective(
                name=o["name"],
                direction=o["direction"],
                unit=o.get("unit", ""),
                target=o.get("target"),
                # ``log_transform`` was added after some rows were
                # already persisted; default to ``False`` for legacy
                # JSON blobs that pre-date the field.
                log_transform=o.get("log_transform", False),
            )
            for o in model.get_objectives()
        ]

        constraints = [
            Constraint(
                type=ConstraintType(c["type"]),
                parameters=c["parameters"],
                value=c["value"],
                coefficients=c.get("coefficients"),
            )
            for c in model.get_constraints()
        ]

        acq_restarts = getattr(model, "acquisition_num_restarts", None)
        acq_samples = getattr(model, "acquisition_raw_samples", None)
        acquisition_optimization: AcquisitionOptimizationConfig | None
        if acq_restarts is None and acq_samples is None:
            acquisition_optimization = None
        else:
            acquisition_optimization = AcquisitionOptimizationConfig(
                num_restarts=acq_restarts,
                raw_samples=acq_samples,
            )

        backend_options_raw = getattr(model, "backend_options_json", None)
        backend_options = json.loads(backend_options_raw) if backend_options_raw else None

        advanced_raw = getattr(model, "advanced_options_json", None)
        advanced = json.loads(advanced_raw) if advanced_raw else {}

        return CampaignSpec(
            name=model.name,
            description=model.description,
            parameters=tuple(parameters),
            objectives=tuple(objectives),
            constraints=tuple(constraints),
            batch_size=model.batch_size,
            max_iterations=model.max_iterations,
            max_observations=getattr(model, "max_observations", None),
            convergence_tolerance=getattr(model, "convergence_tolerance", None),
            acquisition_optimization=acquisition_optimization,
            initial_design_size=model.initial_design_size,
            random_seed=model.random_seed,
            backend=getattr(model, "backend", "botorch"),
            backend_options=backend_options,
            **_advanced_options_kwargs(advanced),
        )


class CampaignRepository:
    """Repository for Campaign entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind this repository to the given async SQLAlchemy session."""
        self.session = session

    async def get(self, entity_id: UUID, *, include_deleted: bool = False) -> Campaign | None:
        """Get campaign by ID.

        Soft-deleted rows are hidden by default; set ``include_deleted=True``
        for admin / forensics queries that need to read historical state
        (see TODO 8.11).
        """
        result = await self.session.execute(
            select(CampaignModel).where(
                CampaignModel.id == str(entity_id),
                *_active_filter(CampaignModel, include_deleted),
            )
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def list_by_owner(
        self, owner_id: UUID, *, include_deleted: bool = False
    ) -> list[Campaign]:
        """List campaigns by owner (hides soft-deleted rows by default)."""
        result = await self.session.execute(
            select(CampaignModel).where(
                CampaignModel.owner_id == str(owner_id),
                *_active_filter(CampaignModel, include_deleted),
            )
        )
        return [self._to_entity(m) for m in result.scalars()]

    async def list_all(self, *, include_deleted: bool = False) -> list[Campaign]:
        """List all campaigns (hides soft-deleted rows by default)."""
        query = select(CampaignModel)
        active = _active_filter(CampaignModel, include_deleted)
        if active:
            query = query.where(*active)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_filtered(
        self,
        owner_id: UUID | None = None,
        status: CampaignStatus | None = None,
        exclude_ids: list[UUID] | None = None,
        limit: int | None = None,
        offset: int = 0,
        *,
        include_deleted: bool = False,
    ) -> tuple[list[Campaign], int]:
        """List campaigns with database-side filtering and pagination.

        Args:
            owner_id: Filter by owner.
            status: Filter by campaign status.
            exclude_ids: Campaign IDs to exclude.
            limit: Maximum number of results.
            offset: Number of results to skip.
            include_deleted: Include soft-deleted rows (admin / forensics only).

        Returns:
            Tuple of (campaigns, total_count).
        """
        query = select(CampaignModel)
        count_query = select(func.count()).select_from(CampaignModel)

        soft_delete_filter = _active_filter(CampaignModel, include_deleted)
        if soft_delete_filter:
            query = query.where(*soft_delete_filter)
            count_query = count_query.where(*soft_delete_filter)

        if owner_id is not None:
            query = query.where(CampaignModel.owner_id == str(owner_id))
            count_query = count_query.where(CampaignModel.owner_id == str(owner_id))
        if status is not None:
            query = query.where(CampaignModel.status == status)
            count_query = count_query.where(CampaignModel.status == status)
        if exclude_ids:
            str_ids = [str(value) for value in exclude_ids]
            query = query.where(CampaignModel.id.notin_(str_ids))
            count_query = count_query.where(CampaignModel.id.notin_(str_ids))

        total_result = await self.session.execute(count_query)
        total_count = total_result.scalar_one()

        # Order matches :meth:`list_keyset` — ``(created_at DESC, id
        # DESC)`` — so a cursor handed back by this offset page chains
        # losslessly when the next page switches to keyset mode. Without
        # the ``id DESC`` tiebreaker the first page can pick an arbitrary
        # order among rows sharing a ``created_at`` value while the
        # keyset comparator uses ``id < cursor_id``, opening a duplicate /
        # skip window on equal-timestamp ties (TODO 8.42 friend-review).
        query = query.order_by(CampaignModel.created_at.desc(), CampaignModel.id.desc())
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def list_keyset(
        self,
        owner_id: UUID | None = None,
        status: CampaignStatus | None = None,
        cursor_created_at: datetime | None = None,
        cursor_id: str | None = None,
        limit: int = 20,
        *,
        include_deleted: bool = False,
    ) -> tuple[list[Campaign], int]:
        """List campaigns via keyset pagination on ``(created_at, id)``.

        The ordering matches :meth:`list_filtered` — newest-first
        descending on ``created_at`` with ``id`` as the descending
        tiebreaker. The cursor walks *backwards* through time: page 1
        returns the newest ``limit`` campaigns, the cursor points at
        the oldest row of that page, and page 2 returns rows strictly
        older than the cursor row. Aligning the two paginators
        guarantees the no-duplicate / no-skip invariant of keyset
        pagination — under the previous ASC-with-``>`` formulation
        the cursor walked the *opposite* direction from page 1 and
        could both duplicate page-1 rows and skip the oldest row
        entirely (TODO 8.42 friend-review finding).

        ``total_count`` is still reported so the agent can show "X of N"
        when desired; under concurrency the totals can drift, which is
        why the cursor (not the offset) is the load-bearing contract.
        """
        base_filters: list[Any] = list(_active_filter(CampaignModel, include_deleted))
        if owner_id is not None:
            base_filters.append(CampaignModel.owner_id == str(owner_id))
        if status is not None:
            base_filters.append(CampaignModel.status == status)

        count_query = select(func.count()).select_from(CampaignModel)
        if base_filters:
            count_query = count_query.where(*base_filters)
        total_count = (await self.session.execute(count_query)).scalar_one()

        query = select(CampaignModel)
        if base_filters:
            query = query.where(*base_filters)
        if cursor_created_at is not None and cursor_id is not None:
            query = query.where(
                or_(
                    CampaignModel.created_at < cursor_created_at,
                    and_(
                        CampaignModel.created_at == cursor_created_at,
                        CampaignModel.id < cursor_id,
                    ),
                )
            )
        query = query.order_by(CampaignModel.created_at.desc(), CampaignModel.id.desc()).limit(
            limit
        )

        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def get_by_ids(
        self, ids: list[UUID], *, include_deleted: bool = False
    ) -> dict[UUID, Campaign]:
        """Get multiple campaigns by their IDs in a single query.

        Args:
            ids: List of UUIDs to fetch.
            include_deleted: Include soft-deleted rows (admin / forensics only).

        Returns:
            Dictionary mapping UUID to Campaign for found campaigns.
        """
        if not ids:
            return {}
        str_ids = [str(value) for value in ids]
        result = await self.session.execute(
            select(CampaignModel).where(
                CampaignModel.id.in_(str_ids),
                *_active_filter(CampaignModel, include_deleted),
            )
        )
        return {UUID(m.id): self._to_entity(m) for m in result.scalars()}

    async def list_recent(
        self,
        owner_id: UUID | None = None,
        limit: int = 5,
        *,
        include_deleted: bool = False,
    ) -> list[Campaign]:
        """Return the most recently created campaigns (newest first).

        Powers the cheap ``campaigns://recent`` discovery resource: a
        bounded, parameter-free listing that an LLM can hit to recover
        from a hallucinated id without paying the cost of paginating
        ``campaigns://list``. ``owner_id`` lets the caller scope to the
        authenticated user; ``limit`` caps the result regardless of how
        many rows the table holds.
        """
        query = select(CampaignModel)
        filters = list(_active_filter(CampaignModel, include_deleted))
        if owner_id is not None:
            filters.append(CampaignModel.owner_id == str(owner_id))
        if filters:
            query = query.where(*filters)
        # ``id DESC`` tiebreaker matches :meth:`list_filtered` and
        # :meth:`list_keyset` so equal-timestamp rows have a single
        # deterministic order across every campaign listing surface.
        query = query.order_by(CampaignModel.created_at.desc(), CampaignModel.id.desc()).limit(
            max(1, limit)
        )
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_active_ids(
        self,
        owner_id: UUID | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[str]:
        """Return string-form campaign ids for fuzzy id-prefix matching.

        Cheap projection (id-only ``SELECT``) so the
        ``CAMPAIGN_NOT_FOUND`` envelope can attach suggestions for a
        typo'd UUID without loading every row.
        """
        query = select(CampaignModel.id)
        filters = list(_active_filter(CampaignModel, include_deleted))
        if owner_id is not None:
            filters.append(CampaignModel.owner_id == str(owner_id))
        if filters:
            query = query.where(*filters)
        result = await self.session.execute(query)
        return [row for (row,) in result.all()]

    async def save(self, campaign: Campaign, expected_version: int | None = None) -> Campaign:
        """Save campaign with optimistic locking + soft-delete guard.

        For new campaigns (first save), ``expected_version`` may be ``None``.
        For updates to existing campaigns, ``expected_version`` **must** be
        provided. The version check and row update happen in a single
        ``UPDATE … WHERE version = expected AND deleted_at IS NULL``
        statement so the operation is atomic even under concurrent
        PostgreSQL connections.

        TODO 8.11 follow-up: ``deleted_at IS NULL`` is part of the OCC
        predicate so a campaign that was soft-deleted between the
        caller's read and this save raises
        :class:`ConcurrentModificationError` instead of silently
        writing fresh state (status, iteration, backend_state,
        hypervolume_history) into a tombstoned row. Without this
        guard, ``generate_suggestions`` phase 3 could leave new
        active suggestions attached to a deleted campaign — the soft-
        delete contract treats the campaign as gone, but the
        downstream INSERTs still happen.
        """
        campaign_id_str = str(campaign.id)
        values = {
            "spec_id": str(campaign.spec_id),
            "owner_id": str(campaign.owner_id),
            "status": campaign.status,
            "version": campaign.version,
            "iteration": campaign.iteration,
            "created_at": campaign.created_at,
            "updated_at": campaign.updated_at,
            "completed_at": campaign.completed_at,
            "turbo_state_json": (
                json.dumps(campaign.backend_state) if campaign.backend_state else None
            ),
            "hypervolume_history_json": json.dumps(campaign.hypervolume_history),
        }

        existing = await self.session.execute(
            select(CampaignModel.id).where(CampaignModel.id == campaign_id_str)
        )
        is_update = existing.scalar_one_or_none() is not None

        if is_update:
            if expected_version is None:
                msg = "Campaign"
                raise ConcurrentModificationError(msg, campaign.id, -1)

            # Atomic UPDATE … WHERE version = expected AND deleted_at IS NULL
            stmt = (
                update(CampaignModel)
                .where(
                    CampaignModel.id == campaign_id_str,
                    CampaignModel.version == expected_version,
                    CampaignModel.deleted_at.is_(None),
                )
                .values(**values)
            )
            result = await self.session.execute(stmt)
            if result.rowcount == 0:  # ty: ignore[unresolved-attribute]
                msg = "Campaign"
                raise ConcurrentModificationError(msg, campaign.id, expected_version)
            await self.session.flush()
        else:
            model = CampaignModel(id=campaign_id_str, **values)
            await self.session.merge(model)
            await self.session.flush()

        return campaign

    async def delete(self, entity_id: UUID) -> bool:
        """Soft-delete a campaign by stamping ``deleted_at``.

        TODO 8.11 replaces hard-delete cascades with a soft-delete
        first-class semantics: the row stays queryable through
        ``include_deleted=True`` so forensics and audit can reconstruct
        history, but normal reads hide it. The suggestion / result
        children stay intact because the FKs are now ``RESTRICT`` — a
        future cleanup that wants to physically remove them must do
        so explicitly via :meth:`hard_delete`.
        """
        stmt = (
            update(CampaignModel)
            .where(CampaignModel.id == str(entity_id), CampaignModel.deleted_at.is_(None))
            .values(deleted_at=utcnow())
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return bool(result.rowcount)  # ty: ignore[unresolved-attribute]

    async def hard_delete(self, entity_id: UUID) -> bool:
        """Physically remove the row (admin / cleanup only).

        Fails with an integrity error when child rows (suggestions,
        results, events) still reference the campaign — the FKs are
        ``ON DELETE RESTRICT`` so accidental cascades cannot wipe
        history. Callers that need a hard delete must soft-delete and
        cascade the children explicitly first.
        """
        result = await self.session.execute(
            select(CampaignModel).where(CampaignModel.id == str(entity_id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    def _to_entity(self, model: CampaignModel) -> Campaign:
        """Convert ORM model to domain entity."""
        return Campaign(
            id=UUID(model.id),
            spec_id=UUID(model.spec_id),
            owner_id=UUID(model.owner_id),
            status=model.status,
            version=model.version,
            iteration=model.iteration,
            created_at=model.created_at,
            updated_at=model.updated_at,
            completed_at=model.completed_at,
            backend_state=model.get_turbo_state(),
            hypervolume_history=model.get_hypervolume_history(),
        )


class SuggestionRepository:
    """Repository for Suggestion entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind this repository to the given async SQLAlchemy session."""
        self.session = session

    async def get(self, entity_id: UUID, *, include_deleted: bool = False) -> Suggestion | None:
        """Get suggestion by ID (hides soft-deleted rows by default)."""
        result = await self.session.execute(
            select(SuggestionModel).where(
                SuggestionModel.id == str(entity_id),
                *_active_filter(SuggestionModel, include_deleted),
            )
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def list_by_campaign(
        self,
        campaign_id: UUID,
        status: SuggestionStatus | None = None,
        *,
        include_deleted: bool = False,
    ) -> list[Suggestion]:
        """List suggestions for a campaign (hides soft-deleted by default)."""
        query = select(SuggestionModel).where(
            SuggestionModel.campaign_id == str(campaign_id),
            *_active_filter(SuggestionModel, include_deleted),
        )
        if status is not None:
            query = query.where(SuggestionModel.status == status)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_actionable_by_campaign(self, campaign_id: UUID) -> list[Suggestion]:
        """List PENDING+ACCEPTED suggestions for a campaign.

        Mirrors :attr:`Suggestion.is_actionable`: both PENDING (generated,
        not yet acknowledged) and ACCEPTED (user-approved, awaiting result)
        suggestions reserve an experiment slot. The budget calculations in
        ``generate_suggestions`` and ``submit_results`` consume this list so
        free-floating submissions and new generations cannot steal slots
        already committed to an actionable suggestion.
        """
        query = select(SuggestionModel).where(
            SuggestionModel.campaign_id == str(campaign_id),
            SuggestionModel.status.in_((SuggestionStatus.PENDING, SuggestionStatus.ACCEPTED)),
            SuggestionModel.deleted_at.is_(None),
        )
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_by_campaign_paginated(
        self,
        campaign_id: UUID,
        status: SuggestionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
        *,
        include_deleted: bool = False,
    ) -> tuple[list[Suggestion], int]:
        """List suggestions with database-level ordering, pagination, and count.

        Returns:
            Tuple of (suggestions, total_count).
        """
        soft_delete_filter = _active_filter(SuggestionModel, include_deleted)
        base = select(SuggestionModel).where(
            SuggestionModel.campaign_id == str(campaign_id),
            *soft_delete_filter,
        )
        count_base = (
            select(func.count())
            .select_from(SuggestionModel)
            .where(
                SuggestionModel.campaign_id == str(campaign_id),
                *soft_delete_filter,
            )
        )
        if status is not None:
            base = base.where(SuggestionModel.status == status)
            count_base = count_base.where(SuggestionModel.status == status)

        total_result = await self.session.execute(count_base)
        total_count = total_result.scalar_one()

        # ``(created_at DESC, id DESC)`` mirrors :meth:`list_by_campaign_keyset`
        # so the two paginators agree even when rows share a
        # ``created_at`` value (see ``CampaignRepository.list_filtered``
        # for the equivalent ordering on campaigns).
        query = (
            base.order_by(SuggestionModel.created_at.desc(), SuggestionModel.id.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def list_by_campaign_keyset(
        self,
        campaign_id: UUID,
        status: SuggestionStatus | None = None,
        cursor_created_at: datetime | None = None,
        cursor_id: str | None = None,
        limit: int = 50,
        *,
        include_deleted: bool = False,
    ) -> tuple[list[Suggestion], int]:
        """List suggestions via keyset pagination on ``(created_at, id)``.

        Ordering matches the offset paginator above (newest-first
        descending). The cursor walks backwards through time so a page
        2 request returns rows strictly older than the cursor row —
        this is the only direction that preserves the no-duplicate /
        no-skip invariant when the first page is newest-first. See
        the corresponding ``CampaignRepository.list_keyset`` docstring
        for the friend-review finding that motivated the realignment.
        """
        base_filters: list[Any] = [
            SuggestionModel.campaign_id == str(campaign_id),
            *_active_filter(SuggestionModel, include_deleted),
        ]
        if status is not None:
            base_filters.append(SuggestionModel.status == status)

        total_count = (
            await self.session.execute(
                select(func.count()).select_from(SuggestionModel).where(*base_filters)
            )
        ).scalar_one()

        query = select(SuggestionModel).where(*base_filters)
        if cursor_created_at is not None and cursor_id is not None:
            query = query.where(
                or_(
                    SuggestionModel.created_at < cursor_created_at,
                    and_(
                        SuggestionModel.created_at == cursor_created_at,
                        SuggestionModel.id < cursor_id,
                    ),
                )
            )
        query = query.order_by(SuggestionModel.created_at.desc(), SuggestionModel.id.desc()).limit(
            limit
        )
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    @staticmethod
    def _suggestion_column_values(suggestion: Suggestion) -> dict[str, Any]:
        """Serialize the Suggestion entity into ORM column values.

        Centralised so the UPDATE and INSERT branches of :meth:`save`
        share the exact same column set.
        """
        return {
            "campaign_id": str(suggestion.campaign_id),
            "parameter_values_json": json.dumps(suggestion.parameter_values),
            "status": suggestion.status,
            "provenance_json": json.dumps(suggestion.provenance.model_dump()),
            "created_at": suggestion.created_at,
            "updated_at": suggestion.updated_at,
        }

    async def save(self, suggestion: Suggestion) -> Suggestion:
        """Save suggestion. Atomic ``deleted_at`` guard on the write itself.

        TODO 8.11 follow-up: the previous implementation read
        ``deleted_at`` first, then merged — a race window where a
        concurrent ``delete()`` landing between the read and the
        merge would silently resurrect the tombstone. The write path
        is now split into an existence check plus either:

        * an atomic ``UPDATE … WHERE id = :id AND deleted_at IS
          NULL`` (raising :class:`ConcurrentModificationError` when
          ``rowcount == 0``, which means the row was tombstoned at
          execution time regardless of what the existence check saw),
          or
        * an ``INSERT`` for genuinely-new rows.

        Suggestions do not carry a version column, so ``-1`` is the
        sentinel ``expected_version`` on the raised CMR — mirroring
        the new-row branch of :meth:`CampaignRepository.save`. A
        dedicated admin restore operation (not yet exposed) is the
        only supported path to clear a tombstone.
        """
        suggestion_id_str = str(suggestion.id)
        existing = await self.session.execute(
            select(SuggestionModel.id).where(SuggestionModel.id == suggestion_id_str)
        )
        values = self._suggestion_column_values(suggestion)

        if existing.scalar_one_or_none() is not None:
            stmt = (
                update(SuggestionModel)
                .where(
                    SuggestionModel.id == suggestion_id_str,
                    SuggestionModel.deleted_at.is_(None),
                )
                .values(**values)
            )
            result = await self.session.execute(stmt)
            if result.rowcount == 0:  # ty: ignore[unresolved-attribute]
                logger.warning(
                    "Refusing to save suggestion %s: row was soft-deleted "
                    "concurrently with this save",
                    suggestion.id,
                )
                msg = "Suggestion"
                raise ConcurrentModificationError(msg, suggestion.id, -1)
        else:
            self.session.add(SuggestionModel(id=suggestion_id_str, deleted_at=None, **values))
        await self.session.flush()
        return suggestion

    async def save_batch(self, suggestions: list[Suggestion]) -> list[Suggestion]:
        """Save multiple suggestions using bulk insert.

        Args:
            suggestions: List of Suggestion entities to save

        Returns:
            List of saved Suggestion entities
        """
        if not suggestions:
            return []

        # Create all ORM models
        models = [
            SuggestionModel(
                id=str(suggestion.id),
                campaign_id=str(suggestion.campaign_id),
                parameter_values_json=json.dumps(suggestion.parameter_values),
                status=suggestion.status,
                provenance_json=json.dumps(suggestion.provenance.model_dump()),
                created_at=suggestion.created_at,
                updated_at=suggestion.updated_at,
            )
            for suggestion in suggestions
        ]

        # Bulk add all models
        self.session.add_all(models)
        await self.session.flush()

        # Return the original entities (they already have the correct data)
        return suggestions

    async def delete(self, entity_id: UUID) -> bool:
        """Soft-delete a suggestion. See :meth:`CampaignRepository.delete`."""
        stmt = (
            update(SuggestionModel)
            .where(SuggestionModel.id == str(entity_id), SuggestionModel.deleted_at.is_(None))
            .values(deleted_at=utcnow())
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return bool(result.rowcount)  # ty: ignore[unresolved-attribute]

    async def hard_delete(self, entity_id: UUID) -> bool:
        """Physically remove the row (admin / cleanup only)."""
        result = await self.session.execute(
            select(SuggestionModel).where(SuggestionModel.id == str(entity_id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    async def transition_status(
        self,
        entity_id: UUID,
        from_status: SuggestionStatus | Sequence[SuggestionStatus],
        to_status: SuggestionStatus,
    ) -> bool:
        """Atomically transition a suggestion from one status to another.

        TODO 8.11 follow-up: status updates used to read the row,
        validate the transition in Python, and write through
        :meth:`save`. Two concurrent transitions from ``PENDING``
        could both pass the in-Python validation and then race to
        overwrite each other (last-writer-wins). This helper folds
        the expected-status guard into the SQL ``UPDATE`` so only
        one of the racing callers wins; the other observes
        ``rowcount == 0`` and can surface a structured conflict.

        ``from_status`` accepts either a single status or a sequence
        — ``submit_results._resolve_suggestion_id`` needs the latter
        to mark either a ``PENDING`` or an ``ACCEPTED`` suggestion
        ``COMPLETED`` in one atomic statement.

        Returns ``True`` when the row was updated. Returns ``False``
        when the row no longer matches the conditional
        (``status ∈ from_status AND deleted_at IS NULL``) —
        typically because a concurrent caller already moved the
        suggestion to a different status or the row was soft-deleted.

        Used by:
        * :func:`update_suggestion_status_operation` for the
          manual PENDING→ACCEPTED / PENDING→REJECTED / ACCEPTED→…
          paths
        * the phase-3 expiration step in ``generate_suggestions``
          (``transition_status(id, PENDING, EXPIRED)``); a count-only
          invariant check cannot detect ``PENDING -> ACCEPTED``
          because both states are actionable
        * ``submit_results._resolve_suggestion_id`` to mark the
          backing suggestion ``COMPLETED`` without races against
          manual status updates
        """
        if isinstance(from_status, SuggestionStatus):
            status_predicate = SuggestionModel.status == from_status
        else:
            status_predicate = SuggestionModel.status.in_(tuple(from_status))
        stmt = (
            update(SuggestionModel)
            .where(
                SuggestionModel.id == str(entity_id),
                status_predicate,
                SuggestionModel.deleted_at.is_(None),
            )
            .values(status=to_status, updated_at=utcnow())
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return bool(result.rowcount)  # ty: ignore[unresolved-attribute]

    async def expire_if_pending(self, entity_id: UUID) -> bool:
        """Atomically mark a still-``PENDING`` suggestion as ``EXPIRED``.

        Convenience wrapper around :meth:`transition_status` —
        ``PENDING -> EXPIRED`` is the only transition the phase-3
        expiration path needs and naming it explicitly keeps the
        call site at that layer readable.
        """
        return await self.transition_status(
            entity_id, SuggestionStatus.PENDING, SuggestionStatus.EXPIRED
        )

    async def list_all(self, *, include_deleted: bool = False) -> list[Suggestion]:
        """List all suggestions (hides soft-deleted by default)."""
        query = select(SuggestionModel)
        active = _active_filter(SuggestionModel, include_deleted)
        if active:
            query = query.where(*active)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def count_pending_by_campaigns(self, campaign_ids: list[UUID]) -> dict[UUID, int]:
        """Count pending suggestions per campaign in a single query.

        Args:
            campaign_ids: List of campaign UUIDs.

        Returns:
            Dictionary mapping campaign UUID to pending suggestion count.
        """
        if not campaign_ids:
            return {}
        str_ids = [str(value) for value in campaign_ids]
        result = await self.session.execute(
            select(SuggestionModel.campaign_id, func.count())
            .where(SuggestionModel.campaign_id.in_(str_ids))
            .where(SuggestionModel.status == SuggestionStatus.PENDING)
            .where(SuggestionModel.deleted_at.is_(None))
            .group_by(SuggestionModel.campaign_id)
        )
        return {UUID(row[0]): row[1] for row in result.all()}

    def _to_entity(self, model: SuggestionModel) -> Suggestion:
        """Convert ORM model to domain entity."""
        prov_data = model.get_provenance()
        provenance = SuggestionProvenance(
            iteration=prov_data["iteration"],
            batch_index=prov_data["batch_index"],
            acquisition_value=prov_data.get("acquisition_value"),
            model_uncertainty=prov_data.get("model_uncertainty"),
            generation_method=prov_data.get("generation_method", "bo"),
            # Enhanced provenance fields
            acquisition_function=prov_data.get("acquisition_function"),
            model_type=prov_data.get("model_type"),
            random_seed=prov_data.get("random_seed"),
            model_version=prov_data.get("model_version"),
            confidence_level=prov_data.get("confidence_level"),
            explanation=prov_data.get("explanation"),
            # Model prediction fields (Step 3)
            predicted_objectives=prov_data.get("predicted_objectives"),
            predicted_std=prov_data.get("predicted_std"),
        )
        return Suggestion(
            id=UUID(model.id),
            campaign_id=UUID(model.campaign_id),
            parameter_values=model.get_parameter_values(),
            status=model.status,
            provenance=provenance,
            created_at=model.created_at,
            updated_at=model.updated_at,
        )


class ResultRepository:
    """Repository for Result entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind this repository to the given async SQLAlchemy session."""
        self.session = session

    async def get(self, entity_id: UUID, *, include_deleted: bool = False) -> Result | None:
        """Get result by ID (hides soft-deleted rows by default)."""
        result = await self.session.execute(
            select(ResultModel).where(
                ResultModel.id == str(entity_id),
                *_active_filter(ResultModel, include_deleted),
            )
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def list_by_campaign(
        self, campaign_id: UUID, *, include_deleted: bool = False
    ) -> list[Result]:
        """List results for a campaign in deterministic insertion order.

        Ordering by ``(created_at, id)`` makes the returned sequence stable
        across query plans, restored database snapshots, and concurrent
        writes — a load-bearing assumption for the convergence-stop
        running-best trajectory and for any future identity-based campaign
        state reconciliation.
        """
        result = await self.session.execute(
            select(ResultModel)
            .where(
                ResultModel.campaign_id == str(campaign_id),
                *_active_filter(ResultModel, include_deleted),
            )
            .order_by(ResultModel.created_at.asc(), ResultModel.id.asc())
        )
        return [self._to_entity(m) for m in result.scalars()]

    async def list_by_campaign_paginated(
        self,
        campaign_id: UUID,
        limit: int = 50,
        offset: int = 0,
        *,
        include_deleted: bool = False,
    ) -> tuple[list[Result], int]:
        """List results with database-level ordering, pagination, and count.

        Returns:
            Tuple of (results, total_count).
        """
        soft_delete_filter = _active_filter(ResultModel, include_deleted)
        base_filters: list[Any] = [
            ResultModel.campaign_id == str(campaign_id),
            *soft_delete_filter,
        ]

        count_result = await self.session.execute(
            select(func.count()).select_from(ResultModel).where(*base_filters)
        )
        total_count = count_result.scalar_one()

        # ``(created_at DESC, id DESC)`` mirrors :meth:`list_by_campaign_keyset`
        # so equal-timestamp rows have a deterministic order that the
        # cursor comparator can chain off (see the campaign repository
        # for the friend-review finding that motivated the tiebreaker).
        query = (
            select(ResultModel)
            .where(*base_filters)
            .order_by(ResultModel.created_at.desc(), ResultModel.id.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def list_by_campaign_keyset(
        self,
        campaign_id: UUID,
        cursor_created_at: datetime | None = None,
        cursor_id: str | None = None,
        limit: int = 50,
        *,
        include_deleted: bool = False,
    ) -> tuple[list[Result], int]:
        """List results via keyset pagination on ``(created_at, id)``.

        Ordering matches :meth:`list_by_campaign_paginated`
        (newest-first descending) so that cursor pages chain correctly
        from the first offset / cursorless page. Walking backwards
        through time with ``< cursor_created_at`` is what gives keyset
        pagination its no-duplicate / no-skip invariant when the
        first page is newest-first; the previous ASC formulation
        could both duplicate page-1 rows and skip the oldest row.
        The non-paginated :meth:`list_by_campaign` retains its
        ``ASC`` ordering for the convergence-trajectory consumer.
        """
        soft_delete_filter = _active_filter(ResultModel, include_deleted)
        base_filters: list[Any] = [
            ResultModel.campaign_id == str(campaign_id),
            *soft_delete_filter,
        ]
        total_count = (
            await self.session.execute(
                select(func.count()).select_from(ResultModel).where(*base_filters)
            )
        ).scalar_one()

        query = select(ResultModel).where(*base_filters)
        if cursor_created_at is not None and cursor_id is not None:
            query = query.where(
                or_(
                    ResultModel.created_at < cursor_created_at,
                    and_(
                        ResultModel.created_at == cursor_created_at,
                        ResultModel.id < cursor_id,
                    ),
                )
            )
        query = query.order_by(ResultModel.created_at.desc(), ResultModel.id.desc()).limit(limit)

        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def list_by_campaigns(
        self, campaign_ids: list[UUID], *, include_deleted: bool = False
    ) -> dict[UUID, list[Result]]:
        """List results for multiple campaigns in a single query.

        Args:
            campaign_ids: List of campaign UUIDs.
            include_deleted: Include soft-deleted rows (admin / forensics only).

        Returns:
            Dictionary mapping campaign UUID to list of results.
        """
        if not campaign_ids:
            return {}
        str_ids = [str(cid) for cid in campaign_ids]
        result = await self.session.execute(
            select(ResultModel).where(
                ResultModel.campaign_id.in_(str_ids),
                *_active_filter(ResultModel, include_deleted),
            )
        )
        by_campaign: dict[UUID, list[Result]] = {cid: [] for cid in campaign_ids}
        for m in result.scalars():
            entity = self._to_entity(m)
            by_campaign[entity.campaign_id].append(entity)
        return by_campaign

    @staticmethod
    def _result_column_values(result: Result) -> dict[str, Any]:
        """Serialize the Result entity into ORM column values.

        Centralised so the UPDATE and INSERT branches of :meth:`save`
        cannot drift apart — every column the entity persists lives
        in exactly one place.
        """
        return {
            "campaign_id": str(result.campaign_id),
            "suggestion_id": str(result.suggestion_id) if result.suggestion_id else None,
            "parameter_values_json": json.dumps(result.parameter_values),
            "objective_values_json": json.dumps(result.objective_values),
            "source": result.source,
            "submitted_by": str(result.submitted_by),
            "measurement_uncertainty_json": (
                json.dumps(result.measurement_uncertainty)
                if result.measurement_uncertainty
                else None
            ),
            "metadata_json": json.dumps(result.metadata),
            "suggestion_snapshot_json": (
                json.dumps(result.suggestion_snapshot)
                if result.suggestion_snapshot is not None
                else None
            ),
            "created_at": result.created_at,
        }

    async def save(self, result: Result) -> Result:
        """Save result. Atomic ``deleted_at`` guard on the write itself.

        TODO 8.11 follow-up: see :meth:`SuggestionRepository.save`
        for the rationale — the previous read-then-merge sequence
        left a race window where a concurrent ``delete()`` between
        the ``deleted_at`` read and the merge could resurrect the
        tombstone. The write path is now split into an existence
        check plus either an atomic
        ``UPDATE … WHERE id = :id AND deleted_at IS NULL``
        (raising :class:`ConcurrentModificationError` on
        ``rowcount == 0``) or an ``INSERT`` for new rows. Results
        are typically write-once, but the symmetric contract keeps
        every mutating repository consistent under concurrent
        deletes.
        """
        result_id_str = str(result.id)
        existing = await self.session.execute(
            select(ResultModel.id).where(ResultModel.id == result_id_str)
        )
        values = self._result_column_values(result)

        if existing.scalar_one_or_none() is not None:
            stmt = (
                update(ResultModel)
                .where(
                    ResultModel.id == result_id_str,
                    ResultModel.deleted_at.is_(None),
                )
                .values(**values)
            )
            response = await self.session.execute(stmt)
            if response.rowcount == 0:  # ty: ignore[unresolved-attribute]
                logger.warning(
                    "Refusing to save result %s: row was soft-deleted concurrently with this save",
                    result.id,
                )
                msg = "Result"
                raise ConcurrentModificationError(msg, result.id, -1)
        else:
            self.session.add(ResultModel(id=result_id_str, deleted_at=None, **values))
        await self.session.flush()
        return result

    async def _fetch_deleted_at(self, result_id: UUID) -> datetime | None:
        """Return the existing row's ``deleted_at`` (None for inserts).

        Kept for tests that need to inspect tombstone state directly.
        The save path no longer relies on this helper — it uses the
        atomic ``UPDATE … WHERE deleted_at IS NULL`` instead.
        """
        row = (
            await self.session.execute(
                select(ResultModel.deleted_at).where(ResultModel.id == str(result_id))
            )
        ).first()
        return row[0] if row is not None else None

    async def save_batch(self, results: list[Result]) -> list[Result]:
        """Save multiple results using bulk insert.

        Args:
            results: List of Result entities to save

        Returns:
            List of saved Result entities
        """
        if not results:
            return []

        # Create all ORM models
        models = [
            ResultModel(
                id=str(r.id),
                campaign_id=str(r.campaign_id),
                suggestion_id=str(r.suggestion_id) if r.suggestion_id else None,
                parameter_values_json=json.dumps(r.parameter_values),
                objective_values_json=json.dumps(r.objective_values),
                source=r.source,
                submitted_by=str(r.submitted_by),
                measurement_uncertainty_json=(
                    json.dumps(r.measurement_uncertainty) if r.measurement_uncertainty else None
                ),
                metadata_json=json.dumps(r.metadata),
                suggestion_snapshot_json=(
                    json.dumps(r.suggestion_snapshot) if r.suggestion_snapshot is not None else None
                ),
                created_at=r.created_at,
            )
            for r in results
        ]

        # Bulk add all models
        self.session.add_all(models)
        await self.session.flush()

        # Return the original entities (they already have the correct data)
        return results

    async def delete(self, entity_id: UUID) -> bool:
        """Soft-delete a result. See :meth:`CampaignRepository.delete`."""
        stmt = (
            update(ResultModel)
            .where(ResultModel.id == str(entity_id), ResultModel.deleted_at.is_(None))
            .values(deleted_at=utcnow())
        )
        result = await self.session.execute(stmt)
        await self.session.flush()
        return bool(result.rowcount)  # ty: ignore[unresolved-attribute]

    async def hard_delete(self, entity_id: UUID) -> bool:
        """Physically remove the row (admin / cleanup only)."""
        result = await self.session.execute(
            select(ResultModel).where(ResultModel.id == str(entity_id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    async def list_all(self, *, include_deleted: bool = False) -> list[Result]:
        """List all results (hides soft-deleted by default)."""
        query = select(ResultModel)
        active = _active_filter(ResultModel, include_deleted)
        if active:
            query = query.where(*active)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def count_by_campaigns(self, campaign_ids: list[UUID]) -> dict[UUID, int]:
        """Count results per campaign in a single query.

        Args:
            campaign_ids: List of campaign UUIDs.

        Returns:
            Dictionary mapping campaign UUID to result count.
        """
        if not campaign_ids:
            return {}
        str_ids = [str(value) for value in campaign_ids]
        result = await self.session.execute(
            select(ResultModel.campaign_id, func.count())
            .where(ResultModel.campaign_id.in_(str_ids))
            .where(ResultModel.deleted_at.is_(None))
            .group_by(ResultModel.campaign_id)
        )
        return {UUID(row[0]): row[1] for row in result.all()}

    def _to_entity(self, model: ResultModel) -> Result:
        """Convert ORM model to domain entity."""
        measurement_uncertainty = None
        if model.measurement_uncertainty_json:
            measurement_uncertainty = json.loads(model.measurement_uncertainty_json)

        return Result(
            id=UUID(model.id),
            campaign_id=UUID(model.campaign_id),
            suggestion_id=UUID(model.suggestion_id) if model.suggestion_id else None,
            parameter_values=model.get_parameter_values(),
            objective_values=model.get_objective_values(),
            source=model.source,
            submitted_by=UUID(model.submitted_by),
            measurement_uncertainty=measurement_uncertainty,
            metadata=model.get_metadata(),
            suggestion_snapshot=model.get_suggestion_snapshot(),
            created_at=model.created_at,
        )


class EventRepository:
    """Repository for audit Event entities."""

    def __init__(self, session: AsyncSession) -> None:
        """Bind this repository to the given async SQLAlchemy session."""
        self.session = session

    async def save(self, event: Event) -> Event:
        """Save an audit event.

        The active workflow ``trace_id`` (if any) is spliced into
        ``input_summary`` so every event-emitting path — the
        ``audit.log_tool_call`` helper, lifecycle / status operations
        that write Events directly — picks it up uniformly. Centralizing
        the splice here means future callers cannot forget it.
        """
        from bo_mcp_server.trace_context import get_trace_id

        enriched_input = dict(event.input_summary)
        trace_id = get_trace_id()
        if trace_id is not None and "trace_id" not in enriched_input:
            enriched_input["trace_id"] = trace_id
        model = EventModel(
            id=str(event.id),
            campaign_id=str(event.campaign_id) if event.campaign_id else None,
            event_type=event.event_type,
            tool_name=event.tool_name,
            input_summary_json=json.dumps(enriched_input),
            output_summary_json=json.dumps(event.output_summary),
            actor_id=event.actor_id,
            created_at=event.created_at,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

    async def list_by_campaign(
        self, campaign_id: UUID, limit: int = 50, *, include_deleted: bool = False
    ) -> list[Event]:
        """List events for a campaign, most recent first."""
        result = await self.session.execute(
            select(EventModel)
            .where(
                EventModel.campaign_id == str(campaign_id),
                *_active_filter(EventModel, include_deleted),
            )
            .order_by(EventModel.created_at.desc())
            .limit(limit)
        )
        return [self._to_entity(m) for m in result.scalars()]

    def _to_entity(self, model: EventModel) -> Event:
        """Convert ORM model to domain entity."""
        return Event(
            id=UUID(model.id),
            campaign_id=UUID(model.campaign_id) if model.campaign_id else None,
            event_type=EventType(model.event_type),
            tool_name=model.tool_name,
            input_summary=json.loads(model.input_summary_json),
            output_summary=json.loads(model.output_summary_json),
            actor_id=model.actor_id,
            created_at=model.created_at,
        )
