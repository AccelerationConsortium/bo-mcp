"""Repository implementations."""

import json
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import (
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
from bo_mcp_server.storage.base import ConcurrentModificationError
from bo_mcp_server.storage.models import (
    CampaignModel,
    CampaignSpecModel,
    EventModel,
    ResultModel,
    SuggestionModel,
    UserModel,
)


class UserRepository:
    """Repository for User entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id: UUID) -> User | None:
        """Get user by ID."""
        result = await self.session.execute(select(UserModel).where(UserModel.id == str(id)))
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

    async def delete(self, id: UUID) -> bool:
        """Delete user by ID."""
        result = await self.session.execute(select(UserModel).where(UserModel.id == str(id)))
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

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id: UUID) -> CampaignSpec | None:
        """Get campaign spec by ID."""
        result = await self.session.execute(
            select(CampaignSpecModel).where(CampaignSpecModel.id == str(id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def save(self, spec: CampaignSpec, spec_id: UUID) -> CampaignSpec:
        """Save campaign spec with explicit ID (specs are immutable)."""
        model = CampaignSpecModel(
            id=str(spec_id),
            name=spec.name,
            description=spec.description,
            parameters_json=json.dumps([p.model_dump() for p in spec.parameters]),
            objectives_json=json.dumps([o.model_dump() for o in spec.objectives]),
            constraints_json=json.dumps([c.model_dump() for c in spec.constraints]),
            batch_size=spec.batch_size,
            max_iterations=spec.max_iterations,
            initial_design_size=spec.initial_design_size,
            random_seed=spec.random_seed,
            backend=spec.backend,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

    async def delete(self, id: UUID) -> bool:
        """Delete campaign spec by ID."""
        result = await self.session.execute(
            select(CampaignSpecModel).where(CampaignSpecModel.id == str(id))
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

        str_ids = [str(id) for id in ids]
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
            )
            for p in model.get_parameters()
        ]

        objectives = [
            Objective(
                name=o["name"],
                direction=o["direction"],
                unit=o.get("unit", ""),
                target=o.get("target"),
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

        return CampaignSpec(
            name=model.name,
            description=model.description,
            parameters=parameters,
            objectives=objectives,
            constraints=constraints,
            batch_size=model.batch_size,
            max_iterations=model.max_iterations,
            initial_design_size=model.initial_design_size,
            random_seed=model.random_seed,
            backend=getattr(model, "backend", "botorch"),
        )


class CampaignRepository:
    """Repository for Campaign entities."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id: UUID) -> Campaign | None:
        """Get campaign by ID."""
        result = await self.session.execute(
            select(CampaignModel).where(CampaignModel.id == str(id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def list_by_owner(self, owner_id: UUID) -> list[Campaign]:
        """List campaigns by owner."""
        result = await self.session.execute(
            select(CampaignModel).where(CampaignModel.owner_id == str(owner_id))
        )
        return [self._to_entity(m) for m in result.scalars()]

    async def list_all(self) -> list[Campaign]:
        """List all campaigns."""
        result = await self.session.execute(select(CampaignModel))
        return [self._to_entity(m) for m in result.scalars()]

    async def list_filtered(
        self,
        owner_id: UUID | None = None,
        status: CampaignStatus | None = None,
        exclude_ids: list[UUID] | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[list[Campaign], int]:
        """List campaigns with database-side filtering and pagination.

        Args:
            owner_id: Filter by owner.
            status: Filter by campaign status.
            exclude_ids: Campaign IDs to exclude.
            limit: Maximum number of results.
            offset: Number of results to skip.

        Returns:
            Tuple of (campaigns, total_count).
        """
        query = select(CampaignModel)
        count_query = select(func.count()).select_from(CampaignModel)

        if owner_id is not None:
            query = query.where(CampaignModel.owner_id == str(owner_id))
            count_query = count_query.where(CampaignModel.owner_id == str(owner_id))
        if status is not None:
            query = query.where(CampaignModel.status == status)
            count_query = count_query.where(CampaignModel.status == status)
        if exclude_ids:
            str_ids = [str(id) for id in exclude_ids]
            query = query.where(CampaignModel.id.notin_(str_ids))
            count_query = count_query.where(CampaignModel.id.notin_(str_ids))

        total_result = await self.session.execute(count_query)
        total_count = total_result.scalar_one()

        query = query.order_by(CampaignModel.created_at.desc())
        if offset > 0:
            query = query.offset(offset)
        if limit is not None:
            query = query.limit(limit)

        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def get_by_ids(self, ids: list[UUID]) -> dict[UUID, Campaign]:
        """Get multiple campaigns by their IDs in a single query.

        Args:
            ids: List of UUIDs to fetch.

        Returns:
            Dictionary mapping UUID to Campaign for found campaigns.
        """
        if not ids:
            return {}
        str_ids = [str(id) for id in ids]
        result = await self.session.execute(
            select(CampaignModel).where(CampaignModel.id.in_(str_ids))
        )
        return {UUID(m.id): self._to_entity(m) for m in result.scalars()}

    async def save(self, campaign: Campaign, expected_version: int | None = None) -> Campaign:
        """Save campaign with optimistic locking.

        For new campaigns (first save), ``expected_version`` may be ``None``.
        For updates to existing campaigns, ``expected_version`` **must** be
        provided.  The version check and row update happen in a single
        ``UPDATE … WHERE version = expected`` statement so the operation is
        atomic even under concurrent PostgreSQL connections.
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
                raise ConcurrentModificationError("Campaign", campaign.id, -1)

            # Atomic UPDATE … WHERE version = expected
            stmt = (
                update(CampaignModel)
                .where(
                    CampaignModel.id == campaign_id_str,
                    CampaignModel.version == expected_version,
                )
                .values(**values)
            )
            result = await self.session.execute(stmt)
            if result.rowcount == 0:  # ty: ignore[unresolved-attribute]
                raise ConcurrentModificationError("Campaign", campaign.id, expected_version)
            await self.session.flush()
        else:
            model = CampaignModel(id=campaign_id_str, **values)
            await self.session.merge(model)
            await self.session.flush()

        return campaign

    async def delete(self, id: UUID) -> bool:
        """Delete campaign by ID."""
        result = await self.session.execute(
            select(CampaignModel).where(CampaignModel.id == str(id))
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

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id: UUID) -> Suggestion | None:
        """Get suggestion by ID."""
        result = await self.session.execute(
            select(SuggestionModel).where(SuggestionModel.id == str(id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def list_by_campaign(
        self, campaign_id: UUID, status: SuggestionStatus | None = None
    ) -> list[Suggestion]:
        """List suggestions for a campaign."""
        query = select(SuggestionModel).where(SuggestionModel.campaign_id == str(campaign_id))
        if status is not None:
            query = query.where(SuggestionModel.status == status)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()]

    async def list_by_campaign_paginated(
        self,
        campaign_id: UUID,
        status: SuggestionStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Suggestion], int]:
        """List suggestions with database-level ordering, pagination, and count.

        Returns:
            Tuple of (suggestions, total_count).
        """
        base = select(SuggestionModel).where(SuggestionModel.campaign_id == str(campaign_id))
        count_base = (
            select(func.count())
            .select_from(SuggestionModel)
            .where(SuggestionModel.campaign_id == str(campaign_id))
        )
        if status is not None:
            base = base.where(SuggestionModel.status == status)
            count_base = count_base.where(SuggestionModel.status == status)

        total_result = await self.session.execute(count_base)
        total_count = total_result.scalar_one()

        query = base.order_by(SuggestionModel.created_at.desc()).offset(offset).limit(limit)
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def save(self, suggestion: Suggestion) -> Suggestion:
        """Save suggestion."""
        model = SuggestionModel(
            id=str(suggestion.id),
            campaign_id=str(suggestion.campaign_id),
            parameter_values_json=json.dumps(suggestion.parameter_values),
            status=suggestion.status,
            provenance_json=json.dumps(suggestion.provenance.model_dump()),
            created_at=suggestion.created_at,
            updated_at=suggestion.updated_at,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

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

    async def delete(self, id: UUID) -> bool:
        """Delete suggestion by ID."""
        result = await self.session.execute(
            select(SuggestionModel).where(SuggestionModel.id == str(id))
        )
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    async def list_all(self) -> list[Suggestion]:
        """List all suggestions."""
        result = await self.session.execute(select(SuggestionModel))
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
        str_ids = [str(id) for id in campaign_ids]
        result = await self.session.execute(
            select(SuggestionModel.campaign_id, func.count())
            .where(SuggestionModel.campaign_id.in_(str_ids))
            .where(SuggestionModel.status == SuggestionStatus.PENDING)
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

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, id: UUID) -> Result | None:
        """Get result by ID."""
        result = await self.session.execute(select(ResultModel).where(ResultModel.id == str(id)))
        model = result.scalar_one_or_none()
        if model is None:
            return None
        return self._to_entity(model)

    async def list_by_campaign(self, campaign_id: UUID) -> list[Result]:
        """List results for a campaign."""
        result = await self.session.execute(
            select(ResultModel).where(ResultModel.campaign_id == str(campaign_id))
        )
        return [self._to_entity(m) for m in result.scalars()]

    async def list_by_campaign_paginated(
        self,
        campaign_id: UUID,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Result], int]:
        """List results with database-level ordering, pagination, and count.

        Returns:
            Tuple of (results, total_count).
        """
        base_where = ResultModel.campaign_id == str(campaign_id)

        count_result = await self.session.execute(
            select(func.count()).select_from(ResultModel).where(base_where)
        )
        total_count = count_result.scalar_one()

        query = (
            select(ResultModel)
            .where(base_where)
            .order_by(ResultModel.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        result = await self.session.execute(query)
        return [self._to_entity(m) for m in result.scalars()], total_count

    async def list_by_campaigns(self, campaign_ids: list[UUID]) -> dict[UUID, list[Result]]:
        """List results for multiple campaigns in a single query.

        Args:
            campaign_ids: List of campaign UUIDs.

        Returns:
            Dictionary mapping campaign UUID to list of results.
        """
        if not campaign_ids:
            return {}
        str_ids = [str(cid) for cid in campaign_ids]
        result = await self.session.execute(
            select(ResultModel).where(ResultModel.campaign_id.in_(str_ids))
        )
        by_campaign: dict[UUID, list[Result]] = {cid: [] for cid in campaign_ids}
        for m in result.scalars():
            entity = self._to_entity(m)
            by_campaign[entity.campaign_id].append(entity)
        return by_campaign

    async def save(self, result: Result) -> Result:
        """Save result."""
        model = ResultModel(
            id=str(result.id),
            campaign_id=str(result.campaign_id),
            suggestion_id=str(result.suggestion_id) if result.suggestion_id else None,
            parameter_values_json=json.dumps(result.parameter_values),
            objective_values_json=json.dumps(result.objective_values),
            source=result.source,
            submitted_by=str(result.submitted_by),
            measurement_uncertainty_json=(
                json.dumps(result.measurement_uncertainty)
                if result.measurement_uncertainty
                else None
            ),
            metadata_json=json.dumps(result.metadata),
            created_at=result.created_at,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

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
                created_at=r.created_at,
            )
            for r in results
        ]

        # Bulk add all models
        self.session.add_all(models)
        await self.session.flush()

        # Return the original entities (they already have the correct data)
        return results

    async def delete(self, id: UUID) -> bool:
        """Delete result by ID."""
        result = await self.session.execute(select(ResultModel).where(ResultModel.id == str(id)))
        model = result.scalar_one_or_none()
        if model is None:
            return False
        await self.session.delete(model)
        return True

    async def list_all(self) -> list[Result]:
        """List all results."""
        result = await self.session.execute(select(ResultModel))
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
        str_ids = [str(id) for id in campaign_ids]
        result = await self.session.execute(
            select(ResultModel.campaign_id, func.count())
            .where(ResultModel.campaign_id.in_(str_ids))
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
            created_at=model.created_at,
        )


class EventRepository:
    """Repository for audit Event entities."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def save(self, event: Event) -> Event:
        """Save an audit event."""
        model = EventModel(
            id=str(event.id),
            campaign_id=str(event.campaign_id) if event.campaign_id else None,
            event_type=event.event_type,
            tool_name=event.tool_name,
            input_summary_json=json.dumps(event.input_summary),
            output_summary_json=json.dumps(event.output_summary),
            actor_id=event.actor_id,
            created_at=event.created_at,
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

    async def list_by_campaign(self, campaign_id: UUID, limit: int = 50) -> list[Event]:
        """List events for a campaign, most recent first."""
        result = await self.session.execute(
            select(EventModel)
            .where(EventModel.campaign_id == str(campaign_id))
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
