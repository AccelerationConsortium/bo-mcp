"""Tests for the duplicate-suggestion_id preflight in migration ``007_suggestion_unique``.

The migration cannot create the partial-unique index on
``results.suggestion_id`` if duplicate rows already exist (which would
happen on databases that ran the buggy submission path). The upgrade
detects this state up front and aborts with a clear, deterministic error
listing each offending suggestion_id and its row count, so the operator
can clean up before re-running.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path

import pytest
from sqlalchemy import Column, MetaData, String, Table, Text, create_engine, insert

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "20260512_000002_add_results_suggestion_id_unique_index.py"
)


def _load_migration():
    """Import the migration module by path so we can call its functions in-process."""
    spec = importlib.util.spec_from_file_location("migration_007", MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["migration_007"] = module
    spec.loader.exec_module(module)
    return module


def _build_results_table(metadata: MetaData) -> Table:
    """Minimal ``results`` table shape that matches the pre-migration schema.

    We only need the columns the preflight query touches; other columns
    can be NULL or omitted. Defining the table here (instead of importing
    the ORM model) keeps the test isolated from later schema changes that
    would otherwise pull in columns added by future migrations.
    """
    return Table(
        "results",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("campaign_id", String(36), nullable=False),
        Column("suggestion_id", String(36), nullable=True),
        Column("parameter_values_json", Text, nullable=False),
        Column("objective_values_json", Text, nullable=False),
        Column("source", String(32), nullable=False),
        Column("submitted_by", String(36), nullable=False),
        Column("metadata_json", Text, nullable=False),
    )


def _row(suggestion_id: str | None) -> dict[str, object]:
    return {
        "id": str(uuid.uuid4()),
        "campaign_id": str(uuid.uuid4()),
        "suggestion_id": suggestion_id,
        "parameter_values_json": "{}",
        "objective_values_json": "{}",
        "source": "api",
        "submitted_by": str(uuid.uuid4()),
        "metadata_json": "{}",
    }


def _patch_alembic_op(monkeypatch, bind, create_index_calls: list[dict]) -> None:
    """Replace ``alembic.op`` so the upgrade can run against a SQLAlchemy bind.

    The migration module uses ``op.get_bind()`` for the preflight query and
    ``op.create_index(...)`` to create the actual constraint. We stub both
    so we can run ``upgrade()`` inside a real test database without booting
    Alembic.
    """
    migration_module = sys.modules["migration_007"]

    class _StubOp:
        @staticmethod
        def get_bind():
            return bind

        @staticmethod
        def create_index(*args, **kwargs):
            create_index_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(migration_module, "op", _StubOp)


class TestPreflight:
    """The upgrade aborts when duplicate ``suggestion_id`` rows exist."""

    def test_upgrade_raises_when_duplicates_exist(self, monkeypatch) -> None:
        """A pre-existing duplicate row blocks index creation with a clear error."""
        migration_module = _load_migration()
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        results = _build_results_table(metadata)
        metadata.create_all(engine)

        shared = "00000000-0000-0000-0000-00000000aaaa"
        with engine.begin() as conn:
            conn.execute(
                insert(results),
                [
                    _row(shared),
                    _row(shared),  # duplicate
                    _row(None),  # free-floating; must be ignored by preflight
                ],
            )

        index_calls: list[dict] = []
        with engine.begin() as conn:
            _patch_alembic_op(monkeypatch, conn, index_calls)
            with pytest.raises(migration_module.DuplicateSuggestionIdError) as excinfo:
                migration_module.upgrade()

        assert shared in str(excinfo.value)
        assert "ix_results_suggestion_id_unique" in str(excinfo.value)
        # Crucially: the index was not created. The operator must clean up
        # before re-running.
        assert index_calls == []

    def test_upgrade_creates_index_when_no_duplicates(self, monkeypatch) -> None:
        """Clean database upgrades normally and calls ``create_index``."""
        migration_module = _load_migration()
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        results = _build_results_table(metadata)
        metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(
                insert(results),
                [
                    _row("11111111-1111-1111-1111-111111111111"),
                    _row("22222222-2222-2222-2222-222222222222"),
                    _row(None),  # NULL is fine -- excluded by partial index.
                    _row(None),
                ],
            )

        index_calls: list[dict] = []
        with engine.begin() as conn:
            _patch_alembic_op(monkeypatch, conn, index_calls)
            migration_module.upgrade()  # must not raise

        assert len(index_calls) == 1
        kwargs = index_calls[0]["kwargs"]
        assert kwargs.get("unique") is True
        # SQLite-style partial-where clause must be on the call.
        assert "sqlite_where" in kwargs
