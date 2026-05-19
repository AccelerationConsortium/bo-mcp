"""``CampaignSpecRepository`` round-trips every advanced spec field.

Pre-1.66 (storage extension) the repository only persisted a subset of
:class:`CampaignSpec`. Fields like ``turbo_config``,
``outcome_constraints``, ``use_input_warping``, etc. survived in memory
but were silently dropped on the way to the database — so suggestion
generation after a reload ran with default behavior.

These tests pin save → reload for each advanced field individually plus
one combined case so a regression in the JSON blob serializer fails
loudly.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from bo_mcp_server.domain import (
    AcquisitionMethod,
    CampaignSpec,
    FidelityParameter,
    InputParameter,
    Objective,
    OutcomeConstraint,
    ParameterType,
    SaasboConfig,
    TransferLearningConfig,
    TurboConfig,
)
from bo_mcp_server.storage.models import Base
from bo_mcp_server.storage.repositories import CampaignSpecRepository

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
async def session() -> AsyncGenerator[AsyncSession]:
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async_session = async_sessionmaker(engine, expire_on_commit=True)
    async with async_session() as s:
        yield s
    await engine.dispose()


def _base_spec(**overrides: object) -> CampaignSpec:
    """Build a minimal valid :class:`CampaignSpec` with optional overrides."""
    parameters = (
        InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
        ),
    )
    objectives = (Objective(name="y", direction="minimize"),)
    return CampaignSpec(
        name="Round Trip",
        parameters=parameters,
        objectives=objectives,
        **overrides,  # ty: ignore[invalid-argument-type]
    )


async def _save_and_reload(session: AsyncSession, spec: CampaignSpec) -> CampaignSpec:
    repo = CampaignSpecRepository(session)
    spec_id = uuid4()
    await repo.save(spec, spec_id)
    await session.commit()
    reloaded = await repo.get(spec_id)
    assert reloaded is not None
    return reloaded


class TestAdvancedFieldRoundTrip:
    @pytest.mark.asyncio
    async def test_acquisition_method_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(acquisition_method=AcquisitionMethod.NOISY_EI)
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.acquisition_method == AcquisitionMethod.NOISY_EI

    @pytest.mark.asyncio
    async def test_use_input_warping_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(use_input_warping=True)
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.use_input_warping is True

    @pytest.mark.asyncio
    async def test_use_cost_aware_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(use_cost_aware=True)
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.use_cost_aware is True

    @pytest.mark.asyncio
    async def test_turbo_config_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(turbo_config=TurboConfig(initial_length=0.5))
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.turbo_config is not None
        assert reloaded.turbo_config.initial_length == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_saasbo_config_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(
            saasbo_config=SaasboConfig(warmup_steps=8, num_samples=8, thinning=2),
        )
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.saasbo_config is not None
        assert reloaded.saasbo_config.warmup_steps == 8

    @pytest.mark.asyncio
    async def test_fidelity_parameter_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(
            fidelity_parameter=FidelityParameter(
                name="fidelity",
                bounds=[0.1, 1.0],  # ty: ignore[invalid-argument-type]
                target=1.0,
            ),
        )
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.fidelity_parameter is not None
        assert reloaded.fidelity_parameter.name == "fidelity"
        assert reloaded.fidelity_parameter.target == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_transfer_learning_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(
            transfer_learning=TransferLearningConfig(
                prior_campaign_ids=("abc-123",),
                num_ranking_samples=128,
            ),
        )
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.transfer_learning is not None
        # ``prior_campaign_ids`` is a tuple on the domain model for deep
        # immutability; the round-trip preserves order and contents.
        assert reloaded.transfer_learning.prior_campaign_ids == ("abc-123",)
        assert reloaded.transfer_learning.num_ranking_samples == 128

    @pytest.mark.asyncio
    async def test_outcome_constraints_round_trip(self, session: AsyncSession) -> None:
        spec = _base_spec(
            outcome_constraints=(
                OutcomeConstraint(objective_name="y", threshold=0.5, greater_than=False),
            ),
        )
        reloaded = await _save_and_reload(session, spec)
        assert len(reloaded.outcome_constraints) == 1
        assert reloaded.outcome_constraints[0].threshold == pytest.approx(0.5)
        assert reloaded.outcome_constraints[0].greater_than is False

    @pytest.mark.asyncio
    async def test_combined_advanced_fields_round_trip(self, session: AsyncSession) -> None:
        """Every advanced field set together must survive together."""
        spec = _base_spec(
            acquisition_method=AcquisitionMethod.NOISY_EI,
            use_input_warping=True,
            use_cost_aware=True,
            turbo_config=TurboConfig(),
            saasbo_config=SaasboConfig(warmup_steps=8, num_samples=8, thinning=2),
            outcome_constraints=(OutcomeConstraint(objective_name="y", threshold=0.0),),
            backend_options={"botorch": {"acquisition_optimizer": "lbfgsb"}},
        )
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.acquisition_method == AcquisitionMethod.NOISY_EI
        assert reloaded.use_input_warping is True
        assert reloaded.use_cost_aware is True
        assert reloaded.turbo_config is not None
        assert reloaded.saasbo_config is not None
        assert len(reloaded.outcome_constraints) == 1
        assert reloaded.backend_options == {"botorch": {"acquisition_optimizer": "lbfgsb"}}

    @pytest.mark.asyncio
    async def test_objective_log_transform_round_trip(self, session: AsyncSession) -> None:
        """``Objective.log_transform`` survives save → reload.

        Reference: BoTorch documents the ``Log → Standardize`` outcome
        stack for multi-decade objectives —
        https://botorch.readthedocs.io/en/stable/models.html#botorch.models.transforms.outcome.Log.
        If the flag is dropped on load, suggestion generation runs
        against a plain Standardize model whenever a campaign is
        revisited after a process restart, silently regressing the
        scale-invariance the flag was supposed to provide.
        """
        spec = CampaignSpec(
            name="Log Transform Round Trip",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="rate", direction="minimize", log_transform=True),),
        )
        reloaded = await _save_and_reload(session, spec)
        assert reloaded.objectives[0].log_transform is True

    @pytest.mark.asyncio
    async def test_objective_log_transform_defaults_to_false_for_legacy_rows(
        self, session: AsyncSession
    ) -> None:
        """Rows persisted before the field existed still load cleanly.

        Simulates a legacy row by writing the objective JSON without
        ``log_transform`` and asserting the reload normalizes to
        ``False`` rather than raising on the missing key.
        """
        from sqlalchemy import update

        from bo_mcp_server.storage.models import CampaignSpecModel
        from bo_mcp_server.storage.repositories import CampaignSpecRepository

        spec = _base_spec()
        repo = CampaignSpecRepository(session)
        spec_id = uuid4()
        await repo.save(spec, spec_id)
        await session.commit()

        # Strip the ``log_transform`` key to emulate a legacy row.
        import json

        legacy = json.dumps(
            [
                {
                    "name": "y",
                    "direction": "minimize",
                    "unit": "",
                    "target": None,
                }
            ]
        )
        await session.execute(
            update(CampaignSpecModel)
            .where(CampaignSpecModel.id == str(spec_id))
            .values(objectives_json=legacy)
        )
        await session.commit()
        reloaded = await repo.get(spec_id)
        assert reloaded is not None
        assert reloaded.objectives[0].log_transform is False

    @pytest.mark.asyncio
    async def test_default_spec_keeps_blob_null(self, session: AsyncSession) -> None:
        """Specs with no advanced fields don't bloat storage with default JSON."""
        from sqlalchemy import select

        from bo_mcp_server.storage.models import CampaignSpecModel

        spec = _base_spec()
        repo = CampaignSpecRepository(session)
        spec_id = uuid4()
        await repo.save(spec, spec_id)
        await session.commit()

        result = await session.execute(
            select(CampaignSpecModel).where(CampaignSpecModel.id == str(spec_id))
        )
        model = result.scalar_one()
        assert model.advanced_options_json is None
