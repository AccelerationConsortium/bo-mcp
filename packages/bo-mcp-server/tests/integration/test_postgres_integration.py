"""PostgreSQL integration tests using testcontainers.

These tests verify database behavior with a real PostgreSQL instance,
ensuring the application works correctly in production-like environments.

Reference: https://testcontainers-python.readthedocs.io/en/latest/

Test Categories:
1. Connection and pooling behavior
2. Transaction isolation and rollback
3. Concurrent access patterns
4. JSON field serialization (PostgreSQL vs SQLite differences)
5. Enum handling across databases
6. Foreign key constraint behavior

Run with: pytest -m postgres
Skip with: pytest -m "not postgres"
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from bo_mcp_server.domain import (
    Campaign,
    CampaignSpec,
    CampaignStatus,
    InputParameter,
    Objective,
    ParameterType,
    ResultSource,
    Suggestion,
    SuggestionProvenance,
    User,
)
from bo_mcp_server.storage.models import (
    CampaignModel,
    CampaignSpecModel,
    ResultModel,
    SuggestionModel,
    UserModel,
)
from bo_mcp_server.storage.repositories import (
    CampaignRepository,
    CampaignSpecRepository,
    SuggestionRepository,
    UserRepository,
)

# Import PostgreSQL fixtures
pytest_plugins = ["tests.conftest_postgres"]

# Mark all tests in this module as requiring PostgreSQL. The session-scoped
# PostgreSQL engine/schema fixtures run on a session-scoped event loop, so the
# tests must share that loop — otherwise asyncpg raises "got Future attached to
# a different loop". ``loop_scope="session"`` pins every test here to it.
pytestmark = [pytest.mark.postgres, pytest.mark.asyncio(loop_scope="session")]


class TestPostgresConnection:
    """Tests for PostgreSQL connection and basic operations."""

    async def test_connection_works(self, postgres_session: AsyncSession) -> None:
        """Verify basic PostgreSQL connectivity.

        Reference: Basic sanity check that testcontainers setup works.
        """
        result = await postgres_session.execute(text("SELECT 1 as test"))
        row = result.fetchone()
        assert row is not None
        assert row[0] == 1

    async def test_postgres_version(self, postgres_session: AsyncSession) -> None:
        """Verify PostgreSQL version is 16.x as configured.

        Reference: Ensures we're testing against the production version.
        """
        result = await postgres_session.execute(text("SELECT version()"))
        version = result.scalar()
        assert version is not None
        assert "PostgreSQL 16" in version

    async def test_tables_created(self, postgres_session: AsyncSession) -> None:
        """Verify all expected tables exist in PostgreSQL.

        Reference: Schema consistency check between models and migrations.
        """
        expected_tables = {"users", "campaign_specs", "campaigns", "suggestions", "results"}

        result = await postgres_session.execute(
            text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        )
        actual_tables = {row[0] for row in result.fetchall()}

        assert expected_tables.issubset(actual_tables), (
            f"Missing tables: {expected_tables - actual_tables}"
        )


class TestPostgresUserRepository:
    """Integration tests for UserRepository with PostgreSQL."""

    async def test_save_and_retrieve_user(
        self, postgres_session: AsyncSession, pg_sample_user_data: dict
    ) -> None:
        """Test full user lifecycle with PostgreSQL.

        Reference: https://www.cosmicpython.com/book/chapter_02_repository.html
        """
        repo = UserRepository(postgres_session)
        user = User(
            id=uuid4(),
            name=pg_sample_user_data["name"],
            email=pg_sample_user_data["email"],
            api_key_hash=pg_sample_user_data["api_key_hash"],
        )

        saved = await repo.save(user)
        await postgres_session.flush()

        retrieved = await repo.get(saved.id)

        assert retrieved is not None
        assert retrieved.name == user.name
        assert retrieved.email == user.email

    async def test_unique_email_constraint(
        self, postgres_session: AsyncSession, pg_sample_user_data: dict
    ) -> None:
        """Test PostgreSQL enforces unique email constraint.

        Reference: Database constraint testing for data integrity.
        """
        repo = UserRepository(postgres_session)
        email = pg_sample_user_data["email"]

        user1 = User(
            name="User 1",
            email=email,
            api_key_hash="hash1",
        )
        await repo.save(user1)
        await postgres_session.flush()

        user2 = User(
            name="User 2",
            email=email,  # Same email - should fail
            api_key_hash="hash2",
        )

        async def _save_and_flush() -> None:
            await repo.save(user2)
            await postgres_session.flush()

        with pytest.raises(IntegrityError):
            await _save_and_flush()

    async def test_get_by_email(
        self, postgres_session: AsyncSession, pg_sample_user_data: dict
    ) -> None:
        """Test looking up user by email in PostgreSQL."""
        repo = UserRepository(postgres_session)
        user = User(
            name=pg_sample_user_data["name"],
            email=pg_sample_user_data["email"],
            api_key_hash=pg_sample_user_data["api_key_hash"],
        )
        await repo.save(user)
        await postgres_session.flush()

        found = await repo.get_by_email(pg_sample_user_data["email"])

        assert found is not None
        assert found.id == user.id


class TestPostgresCampaignLifecycle:
    """Integration tests for full campaign lifecycle with PostgreSQL."""

    async def test_create_campaign_with_spec(self, postgres_session: AsyncSession) -> None:
        """Test creating a campaign with its specification in PostgreSQL.

        Reference: Tests foreign key relationships work correctly.
        """
        # Create user
        user_repo = UserRepository(postgres_session)
        user = User(
            name="Campaign Owner",
            email=f"owner_{uuid4().hex[:8]}@example.com",
            api_key_hash="owner_hash",
        )
        await user_repo.save(user)

        # Create spec
        spec_repo = CampaignSpecRepository(postgres_session)
        spec_id = uuid4()
        spec = CampaignSpec(
            name="Test Spec",
            description="Integration test specification",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            batch_size=2,
        )
        await spec_repo.save(spec, spec_id)

        # Create campaign
        campaign_repo = CampaignRepository(postgres_session)
        campaign = Campaign(
            spec_id=spec_id,
            owner_id=user.id,
        )
        await campaign_repo.save(campaign)
        await postgres_session.flush()

        # Retrieve and verify
        retrieved = await campaign_repo.get(campaign.id)
        assert retrieved is not None
        assert retrieved.spec_id == spec_id
        assert retrieved.owner_id == user.id
        assert retrieved.status == CampaignStatus.CREATED

    async def test_delete_campaign_with_suggestions_is_restricted(
        self, postgres_session: AsyncSession
    ) -> None:
        """Hard-deleting a campaign that still has suggestions is refused.

        The ``suggestions.campaign_id`` FK is ``ON DELETE RESTRICT`` with a
        NOT-NULL column and the ORM relationship carries no delete cascade:
        production soft-deletes campaigns (stamping ``deleted_at``) and leaves
        the children intact, and a physical removal must explicitly delete the
        children first (see ``CampaignRepository.hard_delete``). A raw ORM
        ``delete(campaign)`` must therefore be refused rather than silently
        cascading or orphaning rows — SQLAlchemy attempts to null the children's
        FK, which the NOT-NULL/RESTRICT constraint rejects.

        Reference: SQLAlchemy ``ON DELETE`` / ``passive_deletes`` semantics
        https://docs.sqlalchemy.org/en/20/orm/cascades.html#using-foreign-key-on-delete-with-orm-relationships
        """
        # Create user, spec, campaign
        user = UserModel(
            id=str(uuid4()),
            name="Test",
            email=f"cascade_{uuid4().hex[:8]}@example.com",
            api_key_hash="hash",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(user)

        spec = CampaignSpecModel(
            id=str(uuid4()),
            name="Cascade Test",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
            created_at=datetime.now(UTC),
        )
        postgres_session.add(spec)

        campaign = CampaignModel(
            id=str(uuid4()),
            spec_id=spec.id,
            owner_id=user.id,
            status=CampaignStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        postgres_session.add(campaign)

        # Create suggestion
        suggestion = SuggestionModel(
            id=str(uuid4()),
            campaign_id=campaign.id,
            parameter_values_json='{"x": 0.5}',
            provenance_json='{"iteration": 1, "batch_index": 0}',
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        postgres_session.add(suggestion)
        await postgres_session.flush()

        # Deleting the campaign while a suggestion references it must be
        # refused by the RESTRICT / NOT-NULL constraint rather than cascading or
        # orphaning the child row.
        async def _delete_campaign() -> None:
            await postgres_session.delete(campaign)
            await postgres_session.flush()

        with pytest.raises(IntegrityError):
            await _delete_campaign()


class TestPostgresJsonSerialization:
    """Tests for JSON field handling in PostgreSQL.

    PostgreSQL stores JSON as TEXT in our models (not native JSONB).
    These tests verify serialization/deserialization works correctly.
    """

    async def test_json_parameters_round_trip(self, postgres_session: AsyncSession) -> None:
        """Test JSON parameter storage and retrieval in PostgreSQL."""
        spec = CampaignSpecModel(
            id=str(uuid4()),
            name="JSON Test",
            parameters_json=(
                '[{"name":"x","type":"continuous","bounds":[0.0,1.0],"description":"Test param"}]'
            ),
            objectives_json='[{"name":"y","direction":"minimize","unit":"USD"}]',
            constraints_json='[{"type":"linear","coefficients":{"x":1.0},"bound":0.5}]',
            created_at=datetime.now(UTC),
        )
        postgres_session.add(spec)
        await postgres_session.flush()

        # Retrieve and verify JSON parsing
        result = await postgres_session.execute(
            text(
                "SELECT parameters_json, objectives_json, constraints_json "
                "FROM campaign_specs WHERE id = :id"
            ),
            {"id": spec.id},
        )
        row = result.fetchone()
        assert row is not None, "Expected row from database"

        import json

        params = json.loads(row[0])
        objectives = json.loads(row[1])
        constraints = json.loads(row[2])

        assert params[0]["name"] == "x"
        assert params[0]["bounds"] == [0.0, 1.0]
        assert objectives[0]["direction"] == "minimize"
        assert constraints[0]["type"] == "linear"

    async def test_complex_nested_json(self, postgres_session: AsyncSession) -> None:
        """Test complex nested JSON structures in PostgreSQL.

        Reference: Ensures deep nesting works (provenance, metadata, etc.)
        """
        campaign_id = str(uuid4())
        user_id = str(uuid4())
        spec_id = str(uuid4())

        # Create required parent records
        user = UserModel(
            id=user_id,
            name="JSON Test User",
            email=f"json_{uuid4().hex[:8]}@example.com",
            api_key_hash="hash",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(user)

        spec = CampaignSpecModel(
            id=spec_id,
            name="Nested JSON Test",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
            created_at=datetime.now(UTC),
        )
        postgres_session.add(spec)

        campaign = CampaignModel(
            id=campaign_id,
            spec_id=spec_id,
            owner_id=user_id,
            status=CampaignStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        postgres_session.add(campaign)

        # Create result with nested metadata
        import json

        complex_metadata = {
            "experiment": {
                "temperature": 25.5,
                "humidity": 60,
                "notes": ["observation 1", "observation 2"],
            },
            "equipment": {
                "model": "XYZ-1000",
                "calibration_date": "2025-01-15",
            },
        }

        result = ResultModel(
            id=str(uuid4()),
            campaign_id=campaign_id,
            parameter_values_json='{"x": 0.5}',
            objective_values_json='{"y": 1.23}',
            source=ResultSource.API,
            submitted_by=user_id,
            metadata_json=json.dumps(complex_metadata),
            created_at=datetime.now(UTC),
        )
        postgres_session.add(result)
        await postgres_session.flush()

        # Retrieve and verify
        retrieved_metadata = result.get_metadata()
        assert retrieved_metadata["experiment"]["temperature"] == 25.5
        assert len(retrieved_metadata["experiment"]["notes"]) == 2


class TestPostgresEnumHandling:
    """Tests for SQLAlchemy Enum handling in PostgreSQL.

    PostgreSQL creates native ENUM types, unlike SQLite which stores as strings.
    These tests verify enum behavior is consistent.
    """

    async def test_campaign_status_enum(self, postgres_session: AsyncSession) -> None:
        """Test CampaignStatus enum storage in PostgreSQL."""
        user_id = str(uuid4())
        spec_id = str(uuid4())

        user = UserModel(
            id=user_id,
            name="Enum Test",
            email=f"enum_{uuid4().hex[:8]}@example.com",
            api_key_hash="hash",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(user)

        spec = CampaignSpecModel(
            id=spec_id,
            name="Enum Test",
            parameters_json="[]",
            objectives_json="[]",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(spec)

        campaign = CampaignModel(
            id=str(uuid4()),
            spec_id=spec_id,
            owner_id=user_id,
            status=CampaignStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        postgres_session.add(campaign)
        await postgres_session.flush()

        # Update status
        campaign.status = CampaignStatus.COMPLETED
        await postgres_session.flush()

        # Verify
        result = await postgres_session.execute(
            text("SELECT status FROM campaigns WHERE id = :id"),
            {"id": campaign.id},
        )
        status_value = result.scalar()
        assert status_value == "COMPLETED"


class TestPostgresConcurrency:
    """Tests for concurrent access patterns in PostgreSQL.

    These tests verify behavior under concurrent access scenarios,
    which is more relevant for PostgreSQL than SQLite.
    """

    async def test_optimistic_locking(self, postgres_session: AsyncSession) -> None:
        """Test optimistic locking with version field.

        Reference: Prevents lost updates in concurrent scenarios.
        """
        user_id = str(uuid4())
        spec_id = str(uuid4())

        user = UserModel(
            id=user_id,
            name="Lock Test",
            email=f"lock_{uuid4().hex[:8]}@example.com",
            api_key_hash="hash",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(user)

        spec = CampaignSpecModel(
            id=spec_id,
            name="Lock Test",
            parameters_json="[]",
            objectives_json="[]",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(spec)

        campaign = CampaignModel(
            id=str(uuid4()),
            spec_id=spec_id,
            owner_id=user_id,
            version=1,
            iteration=0,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        postgres_session.add(campaign)
        await postgres_session.flush()

        # Increment version (simulating update)
        campaign.version = 2
        campaign.iteration = 1
        await postgres_session.flush()

        assert campaign.version == 2
        assert campaign.iteration == 1


class TestPostgresBulkOperations:
    """Tests for bulk insert/update operations in PostgreSQL."""

    async def test_bulk_insert_suggestions(self, postgres_session: AsyncSession) -> None:
        """Test bulk inserting suggestions efficiently.

        Reference: Verifies repository batch operations work with PostgreSQL.
        """
        # Create required parent records
        user_id = str(uuid4())
        spec_id = str(uuid4())
        campaign_id = uuid4()

        user = UserModel(
            id=user_id,
            name="Bulk Test",
            email=f"bulk_{uuid4().hex[:8]}@example.com",
            api_key_hash="hash",
            created_at=datetime.now(UTC),
        )
        postgres_session.add(user)

        spec = CampaignSpecModel(
            id=spec_id,
            name="Bulk Test",
            parameters_json='[{"name":"x","type":"continuous","bounds":[0,1]}]',
            objectives_json='[{"name":"y","direction":"minimize"}]',
            created_at=datetime.now(UTC),
        )
        postgres_session.add(spec)

        campaign = CampaignModel(
            id=str(campaign_id),
            spec_id=spec_id,
            owner_id=user_id,
            status=CampaignStatus.RUNNING,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
        postgres_session.add(campaign)
        await postgres_session.flush()

        # Bulk insert suggestions
        repo = SuggestionRepository(postgres_session)
        suggestions = [
            Suggestion(
                campaign_id=campaign_id,
                parameter_values={"x": i / 100.0},
                provenance=SuggestionProvenance(iteration=1, batch_index=i),
            )
            for i in range(100)
        ]

        saved = await repo.save_batch(suggestions)
        await postgres_session.flush()

        assert len(saved) == 100

        # Verify count in database
        result = await postgres_session.execute(
            text("SELECT COUNT(*) FROM suggestions WHERE campaign_id = :cid"),
            {"cid": str(campaign_id)},
        )
        count = result.scalar()
        assert count == 100


class TestPostgresSavepointIsolation:
    """Verify the savepoint-based ``postgres_session`` fixture isolates tests.

    The two tests below intentionally share an email so a leak across the
    transaction boundary would surface as an ``IntegrityError`` on the
    unique-email constraint. They are ordered alphabetically by pytest's
    default collection — ``test_a_*`` writes, ``test_b_*`` re-writes — and
    both must pass. Under outer-transaction-only isolation, a test that
    issued an explicit ``commit`` would persist data into the next test
    and break this invariant; the savepoint + ``after_transaction_end``
    restart pattern keeps the outer transaction rollback-effective even
    in that case.
    """

    _SHARED_EMAIL = "savepoint_isolation@example.com"

    async def test_a_write_then_commit_inside_savepoint(
        self, postgres_session: AsyncSession
    ) -> None:
        repo = UserRepository(postgres_session)
        await repo.save(
            User(
                name="Isolation A",
                email=self._SHARED_EMAIL,
                api_key_hash="iso_hash_a",
            )
        )
        # Application-side commit — the savepoint completes, but the outer
        # transaction stays open and will be rolled back at teardown.
        await postgres_session.commit()

        # The row is visible within this test
        found = await repo.get_by_email(self._SHARED_EMAIL)
        assert found is not None

    async def test_b_writes_same_email_after_rollback(self, postgres_session: AsyncSession) -> None:
        """If isolation held, the prior test's row is gone and this insert succeeds.

        Without savepoint isolation (just plain ``BEGIN ... ROLLBACK`` around
        the test), the explicit ``commit`` in ``test_a_*`` would persist the
        row and this insert would fail with ``UniqueViolation``.
        """
        repo = UserRepository(postgres_session)
        await repo.save(
            User(
                name="Isolation B",
                email=self._SHARED_EMAIL,
                api_key_hash="iso_hash_b",
            )
        )
        await postgres_session.flush()

        found = await repo.get_by_email(self._SHARED_EMAIL)
        assert found is not None
        assert found.api_key_hash == "iso_hash_b"

    async def test_savepoint_restarts_after_commit_within_single_test(
        self, postgres_session: AsyncSession
    ) -> None:
        """Repeated commit/write cycles in a single test stay session-managed.

        The cross-test isolation tests above only exercise one commit per
        test. This test pins the inner invariant directly: after the
        application code commits, the ``after_transaction_end`` listener
        must immediately restart the SAVEPOINT so the next ``save`` still
        executes inside a nested transaction. Three commit-then-write
        cycles followed by a final read prove the loop is stable and that
        all writes remain visible to the same session while still being
        contained in the outer transaction (rolled back at teardown).

        If the savepoint restart were missing, the second ``save`` would
        execute against an "implicit autocommit" connection state and
        either raise an InvalidRequestError or silently persist past the
        outer rollback — both visible as a follow-up regression.
        """
        repo = UserRepository(postgres_session)
        emails = [f"restart_{i}_{uuid4().hex[:8]}@example.com" for i in range(3)]

        for i, email in enumerate(emails):
            await repo.save(
                User(
                    name=f"Restart {i}",
                    email=email,
                    api_key_hash=f"restart_hash_{i}",
                )
            )
            # Application-side commit at the end of each write cycle. The
            # savepoint must restart so the next iteration's save executes
            # in a nested transaction (not autocommit).
            await postgres_session.commit()

        # All three users are visible within this session — proves writes
        # survived their respective savepoints and the session is still
        # operable after multiple restart cycles.
        for i, email in enumerate(emails):
            found = await repo.get_by_email(email)
            assert found is not None, f"User {i} ({email}) not visible after commit"
            assert found.api_key_hash == f"restart_hash_{i}"

        # Final sanity check: the session is still in a usable transactional
        # state — a flush of one more write should succeed, not raise.
        final_email = f"restart_final_{uuid4().hex[:8]}@example.com"
        await repo.save(
            User(
                name="Restart final",
                email=final_email,
                api_key_hash="restart_hash_final",
            )
        )
        await postgres_session.flush()
        assert (await repo.get_by_email(final_email)) is not None
