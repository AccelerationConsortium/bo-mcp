"""Repository implementations."""

import json
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import (
    Campaign,
    CampaignSpec,
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
from bo_mcp_server.storage.base import ConcurrentModificationError
from bo_mcp_server.storage.models import (
    CampaignModel,
    CampaignSpecModel,
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

    async def save(self, campaign: Campaign, expected_version: int | None = None) -> Campaign:
        """Save campaign with optional optimistic locking."""
        if expected_version is not None:
            # Check version for optimistic locking
            existing = await self.session.execute(
                select(CampaignModel).where(CampaignModel.id == str(campaign.id))
            )
            existing_model = existing.scalar_one_or_none()
            if existing_model and existing_model.version != expected_version:
                raise ConcurrentModificationError("Campaign", campaign.id, expected_version)

        model = CampaignModel(
            id=str(campaign.id),
            spec_id=str(campaign.spec_id),
            owner_id=str(campaign.owner_id),
            status=campaign.status,
            version=campaign.version,
            iteration=campaign.iteration,
            created_at=campaign.created_at,
            updated_at=campaign.updated_at,
            completed_at=campaign.completed_at,
            turbo_state_json=json.dumps(campaign.turbo_state) if campaign.turbo_state else None,
            hypervolume_history_json=json.dumps(campaign.hypervolume_history),
        )
        merged = await self.session.merge(model)
        return self._to_entity(merged)

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
            turbo_state=model.get_turbo_state(),
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

    def _to_entity(self, model: ResultModel) -> Result:
        """Convert ORM model to domain entity."""
        return Result(
            id=UUID(model.id),
            campaign_id=UUID(model.campaign_id),
            suggestion_id=UUID(model.suggestion_id) if model.suggestion_id else None,
            parameter_values=model.get_parameter_values(),
            objective_values=model.get_objective_values(),
            source=model.source,
            submitted_by=UUID(model.submitted_by),
            metadata=model.get_metadata(),
            created_at=model.created_at,
        )
