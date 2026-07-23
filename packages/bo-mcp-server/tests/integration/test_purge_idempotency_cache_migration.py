"""Tests for migration ``017_purge_idempotency_cache``.

Suggestion payloads dropped the ``id`` key in favour of the required
``suggestion_id``, so cached generate responses persisted under the
old shape must not replay. The migration empties the cache; these
tests pin that behaviour so the schema change cannot ship without the
purge, replacing the runtime replay-normalization coverage that was
removed together with the compatibility alias.

Mirrors the in-process pattern of ``test_unique_api_key_hash_migration``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import (
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    insert,
    select,
    text,
)

MIGRATION_PATH = (
    Path(__file__).resolve().parents[2]
    / "migrations"
    / "versions"
    / "20260721_000000_purge_idempotency_cache.py"
)


def _load_migration():
    """Import the migration module by path so we can call its functions in-process."""
    spec = importlib.util.spec_from_file_location("migration_017", MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["migration_017"] = module
    spec.loader.exec_module(module)
    return module


def _build_cache_table(metadata: MetaData) -> Table:
    """Minimal ``idempotency_cache`` shape matching the pre-migration schema."""
    return Table(
        "idempotency_cache",
        metadata,
        Column("tool_name", String(255), primary_key=True),
        Column("idempotency_key", String(255), primary_key=True),
        Column("request_hash", String(64), nullable=False),
        Column("reservation_token", String(36), nullable=True),
        Column("response_json", Text, nullable=False),
        Column("created_at", DateTime(timezone=True), nullable=False),
        Column("expires_at", DateTime(timezone=True), nullable=False),
    )


def _legacy_generate_row() -> dict[str, object]:
    """A cached generate response in the retired id-only payload shape."""
    now = datetime.now(UTC)
    return {
        "tool_name": "bo_generate_suggestions",
        "idempotency_key": str(uuid.uuid4()),
        "request_hash": "a" * 64,
        "reservation_token": None,
        "response_json": json.dumps(
            {
                "success": True,
                "suggestions": [{"id": str(uuid.uuid4()), "parameter_values": {"x": 0.5}}],
                "iteration": 1,
                "errors": [],
            }
        ),
        "created_at": now,
        "expires_at": now + timedelta(hours=24),
    }


def _patch_alembic_op(monkeypatch, bind) -> None:
    """Replace ``alembic.op`` so the upgrade executes against a SQLAlchemy bind."""
    migration_module = sys.modules["migration_017"]

    class _StubOp:
        @staticmethod
        def execute(statement):
            bind.execute(text(statement) if isinstance(statement, str) else statement)

    monkeypatch.setattr(migration_module, "op", _StubOp)


class TestPurgeIdempotencyCache:
    def test_upgrade_deletes_all_cached_rows(self, monkeypatch) -> None:
        """Every cached row is gone after the upgrade, legacy shape included."""
        migration_module = _load_migration()
        engine = create_engine("sqlite:///:memory:")
        metadata = MetaData()
        cache = _build_cache_table(metadata)
        metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(insert(cache), [_legacy_generate_row(), _legacy_generate_row()])

        with engine.begin() as conn:
            _patch_alembic_op(monkeypatch, conn)
            migration_module.upgrade()

        with engine.connect() as conn:
            remaining = conn.execute(select(cache)).fetchall()
        assert remaining == []
