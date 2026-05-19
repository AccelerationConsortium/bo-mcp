"""Tests for the duplicate-``api_key_hash`` preflight in migration ``012_user_api_key_unique``.

The migration replaces the legacy non-unique ``ix_users_api_key_hash``
with a UNIQUE index. If duplicate hashes already exist on disk (e.g.
hand-provisioned dev environments), creating the UNIQUE index would
fail mid-DDL on PostgreSQL, leaving the schema in a half-migrated
state. The upgrade therefore detects the duplicate state up front and
aborts with a deterministic error listing each offending hash and its
row count so the operator can clean up before re-running.

Mirrors the pattern used by ``test_unique_suggestion_id_migration``.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    create_engine,
    insert,
)

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "20260516_000000_add_users_api_key_hash_unique_index.py"
)


def _load_migration():
    """Import the migration module by path so we can call its functions in-process."""
    spec = importlib.util.spec_from_file_location("migration_012", MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["migration_012"] = module
    spec.loader.exec_module(module)
    return module


def _build_users_table(metadata: MetaData) -> Table:
    """Minimal ``users`` table shape that matches the pre-migration schema.

    Mirrors the columns the migration touches; the legacy
    ``ix_users_api_key_hash`` non-unique index is declared explicitly
    so ``upgrade()`` can drop it.
    """
    table = Table(
        "users",
        metadata,
        Column("id", String(36), primary_key=True),
        Column("name", String(255), nullable=False),
        Column("email", String(255), nullable=False, unique=True),
        Column("api_key_hash", String(255), nullable=False),
        Column("is_active", Boolean(), nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("last_active_at", DateTime(timezone=True), nullable=True),
    )
    Index("ix_users_api_key_hash", table.c.api_key_hash)
    return table


def _row(email: str, api_key_hash: str) -> dict[str, object]:
    return {
        "id": str(uuid.uuid4()),
        "name": "test",
        "email": email,
        "api_key_hash": api_key_hash,
        "is_active": True,
        "created_at": datetime.now(UTC),
        "last_active_at": None,
    }


def _patch_alembic_op(
    monkeypatch,
    bind,
    create_index_calls: list[dict],
    drop_index_calls: list[dict],
) -> None:
    """Replace ``alembic.op`` so the upgrade can run against a SQLAlchemy bind."""
    migration_module = sys.modules["migration_012"]

    class _StubOp:
        @staticmethod
        def get_bind():
            return bind

        @staticmethod
        def create_index(*args, **kwargs):
            create_index_calls.append({"args": args, "kwargs": kwargs})

        @staticmethod
        def drop_index(*args, **kwargs):
            drop_index_calls.append({"args": args, "kwargs": kwargs})

    monkeypatch.setattr(migration_module, "op", _StubOp)


class TestPreflight:
    """The upgrade aborts when duplicate ``api_key_hash`` rows exist."""

    def test_upgrade_raises_when_duplicates_exist(self, monkeypatch) -> None:
        """A pre-existing duplicate row blocks index creation with a clear error."""
        migration_module = _load_migration()
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        users = _build_users_table(metadata)
        metadata.create_all(engine)

        shared_hash = "shared-hash"
        with engine.begin() as conn:
            conn.execute(
                insert(users),
                [
                    _row("a@example.com", shared_hash),
                    _row("b@example.com", shared_hash),  # duplicate hash
                    _row("c@example.com", "distinct-hash"),
                ],
            )

        create_calls: list[dict] = []
        drop_calls: list[dict] = []
        with engine.begin() as conn:
            _patch_alembic_op(monkeypatch, conn, create_calls, drop_calls)
            with pytest.raises(migration_module.DuplicateApiKeyHashError) as excinfo:
                migration_module.upgrade()

        assert shared_hash in str(excinfo.value)
        assert "ix_users_api_key_hash_unique" in str(excinfo.value)
        # Crucially: neither the drop nor the create ran. The operator
        # must clean up before re-running.
        assert create_calls == []
        assert drop_calls == []

    def test_upgrade_swaps_indexes_when_no_duplicates(self, monkeypatch) -> None:
        """A clean dataset drops the legacy index and creates the UNIQUE one."""
        migration_module = _load_migration()
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        users = _build_users_table(metadata)
        metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(
                insert(users),
                [
                    _row("a@example.com", "hash-a"),
                    _row("b@example.com", "hash-b"),
                ],
            )

        create_calls: list[dict] = []
        drop_calls: list[dict] = []
        with engine.begin() as conn:
            _patch_alembic_op(monkeypatch, conn, create_calls, drop_calls)
            migration_module.upgrade()  # must not raise

        assert len(drop_calls) == 1
        assert drop_calls[0]["args"][0] == "ix_users_api_key_hash"

        assert len(create_calls) == 1
        assert create_calls[0]["args"][0] == "ix_users_api_key_hash_unique"
        assert create_calls[0]["kwargs"].get("unique") is True
