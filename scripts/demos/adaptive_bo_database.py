#!/usr/bin/env python3
"""SQLite database for Adaptive BO Demo tracking.

This module provides SQLAlchemy ORM models and an async database manager
for tracking optimization runs, iterations, method selections, evaluations,
and model performance metrics.

Usage:
    from adaptive_bo_database import AdaptiveBODatabase

    db = AdaptiveBODatabase(Path("data/adaptive_bo_demo.db"))
    await db.init_schema()

    run_id = await db.create_run(...)
    iteration_id = await db.log_iteration(run_id, 1)
    await db.log_method_selection(iteration_id, {...})
    await db.log_evaluations(iteration_id, [...])
"""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, selectinload


class Base(DeclarativeBase):
    """Base class for adaptive BO ORM models."""


class RunModel(Base):
    """Optimization run metadata."""

    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    benchmark_function: Mapped[str] = mapped_column(String(100), nullable=False)
    n_parameters: Mapped[int] = mapped_column(Integer, nullable=False)
    n_objectives: Mapped[int] = mapped_column(Integer, nullable=False)
    batch_size: Mapped[int] = mapped_column(Integer, nullable=False)
    mcp_campaign_id: Mapped[str] = mapped_column(String(36), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    iterations: Mapped[list["IterationModel"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class IterationModel(Base):
    """Single optimization iteration."""

    __tablename__ = "iterations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(36), ForeignKey("runs.id"), nullable=False)
    iteration_number: Mapped[int] = mapped_column(Integer, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    run: Mapped["RunModel"] = relationship(back_populates="iterations")
    method_selection: Mapped["MethodSelectionModel | None"] = relationship(
        back_populates="iteration", uselist=False, cascade="all, delete-orphan"
    )
    evaluations: Mapped[list["EvaluationModel"]] = relationship(
        back_populates="iteration", cascade="all, delete-orphan"
    )
    model_metrics: Mapped[list["ModelPerformanceModel"]] = relationship(
        back_populates="iteration", cascade="all, delete-orphan"
    )
    best_value: Mapped["BestValueModel | None"] = relationship(
        back_populates="iteration", uselist=False, cascade="all, delete-orphan"
    )


class MethodSelectionModel(Base):
    """Method selection for an iteration (model, acquisition, strategy)."""

    __tablename__ = "method_selections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.id"), nullable=False)
    model_type: Mapped[str] = mapped_column(String(100), nullable=False)
    acquisition_function: Mapped[str] = mapped_column(String(100), nullable=False)
    optimization_strategy: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_transforms_json: Mapped[str] = mapped_column(Text, default="[]")
    alternatives_json: Mapped[str] = mapped_column(Text, default="[]")
    warnings_json: Mapped[str] = mapped_column(Text, default="[]")

    # Relationships
    iteration: Mapped["IterationModel"] = relationship(back_populates="method_selection")

    def get_input_transforms(self) -> list[str]:
        """Get input transforms as list."""
        return json.loads(self.input_transforms_json)

    def get_alternatives(self) -> list[dict[str, str]]:
        """Get alternatives as list of dicts."""
        return json.loads(self.alternatives_json)

    def get_warnings(self) -> list[str]:
        """Get warnings as list."""
        return json.loads(self.warnings_json)


class ModelPerformanceModel(Base):
    """Model performance metrics for an iteration."""

    __tablename__ = "model_performance"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.id"), nullable=False)
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float | None] = mapped_column(Float, nullable=True)
    metric_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    iteration: Mapped["IterationModel"] = relationship(back_populates="model_metrics")

    def get_metric_data(self) -> float | dict[str, Any]:
        """Get metric data, parsing JSON if needed."""
        if self.metric_json:
            return json.loads(self.metric_json)
        return self.metric_value or 0.0


class EvaluationModel(Base):
    """Single evaluation (one point in a batch)."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.id"), nullable=False)
    batch_index: Mapped[int] = mapped_column(Integer, nullable=False)
    parameter_values_json: Mapped[str] = mapped_column(Text, nullable=False)
    objective_values_json: Mapped[str] = mapped_column(Text, nullable=False)
    suggestion_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    iteration: Mapped["IterationModel"] = relationship(back_populates="evaluations")

    def get_parameter_values(self) -> dict[str, float]:
        """Get parameter values as dict."""
        return json.loads(self.parameter_values_json)

    def get_objective_values(self) -> dict[str, float]:
        """Get objective values as dict."""
        return json.loads(self.objective_values_json)


class BestValueModel(Base):
    """Best value found up to this iteration."""

    __tablename__ = "best_values"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    iteration_id: Mapped[int] = mapped_column(Integer, ForeignKey("iterations.id"), nullable=False)
    objective_name: Mapped[str] = mapped_column(String(100), nullable=False)
    best_value: Mapped[float] = mapped_column(Float, nullable=False)
    best_params_json: Mapped[str] = mapped_column(Text, nullable=False)
    is_improvement: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    iteration: Mapped["IterationModel"] = relationship(back_populates="best_value")

    def get_best_params(self) -> dict[str, float]:
        """Get best parameters as dict."""
        return json.loads(self.best_params_json)


class AdaptiveBODatabase:
    """Async SQLAlchemy database manager for adaptive BO tracking."""

    def __init__(self, db_path: Path) -> None:
        """Initialize database manager.

        Args:
            db_path: Path to the SQLite database file.
        """
        self.db_path = db_path
        db_url = f"sqlite+aiosqlite:///{db_path}"
        self.engine = create_async_engine(db_url, echo=False)
        self.session_factory = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

    async def init_schema(self) -> None:
        """Create all tables if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_run(
        self,
        name: str,
        benchmark: str,
        n_params: int,
        n_objectives: int,
        batch_size: int,
        mcp_campaign_id: str,
    ) -> str:
        """Create a new optimization run.

        Args:
            name: Human-readable name for the run.
            benchmark: Name of the benchmark function.
            n_params: Number of parameters.
            n_objectives: Number of objectives.
            batch_size: Batch size for parallel evaluations.
            mcp_campaign_id: Campaign ID from MCP server.

        Returns:
            Generated run ID (UUID).
        """
        run_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            run = RunModel(
                id=run_id,
                name=name,
                benchmark_function=benchmark,
                n_parameters=n_params,
                n_objectives=n_objectives,
                batch_size=batch_size,
                mcp_campaign_id=mcp_campaign_id,
            )
            session.add(run)
            await session.commit()
        return run_id

    async def complete_run(self, run_id: str) -> None:
        """Mark a run as completed.

        Args:
            run_id: ID of the run to complete.
        """
        async with self.session_factory() as session:
            result = await session.execute(select(RunModel).where(RunModel.id == run_id))
            run = result.scalar_one_or_none()
            if run:
                run.completed_at = datetime.now(UTC)
                await session.commit()

    async def log_iteration(self, run_id: str, iteration_number: int) -> int:
        """Log a new iteration.

        Args:
            run_id: ID of the parent run.
            iteration_number: Iteration number (1-indexed).

        Returns:
            Generated iteration ID.
        """
        async with self.session_factory() as session:
            iteration = IterationModel(
                run_id=run_id,
                iteration_number=iteration_number,
            )
            session.add(iteration)
            await session.commit()
            await session.refresh(iteration)
            return iteration.id

    async def log_method_selection(
        self, iteration_id: int, method_selection: dict[str, Any]
    ) -> None:
        """Log method selection for an iteration.

        Args:
            iteration_id: ID of the parent iteration.
            method_selection: Dictionary from generate_suggestions containing
                model_type, acquisition_function, optimization_strategy, etc.
        """
        async with self.session_factory() as session:
            ms = MethodSelectionModel(
                iteration_id=iteration_id,
                model_type=method_selection.get("model_type", "unknown"),
                acquisition_function=method_selection.get("acquisition_function", "unknown"),
                optimization_strategy=method_selection.get("optimization_strategy", "unknown"),
                confidence=method_selection.get("confidence"),
                explanation=method_selection.get("explanation"),
                input_transforms_json=json.dumps(method_selection.get("input_transforms", [])),
                alternatives_json=json.dumps(method_selection.get("alternatives", [])),
                warnings_json=json.dumps(method_selection.get("warnings", [])),
            )
            session.add(ms)
            await session.commit()

    async def log_model_performance(
        self, iteration_id: int, metrics: dict[str, float | dict[str, Any]]
    ) -> None:
        """Log model performance metrics for an iteration.

        Args:
            iteration_id: ID of the parent iteration.
            metrics: Dictionary mapping metric names to values (float or dict).
        """
        async with self.session_factory() as session:
            for name, value in metrics.items():
                if isinstance(value, dict):
                    metric = ModelPerformanceModel(
                        iteration_id=iteration_id,
                        metric_name=name,
                        metric_json=json.dumps(value),
                    )
                else:
                    metric = ModelPerformanceModel(
                        iteration_id=iteration_id,
                        metric_name=name,
                        metric_value=float(value),
                    )
                session.add(metric)
            await session.commit()

    async def log_evaluations(self, iteration_id: int, evaluations: list[dict[str, Any]]) -> None:
        """Log batch evaluations for an iteration.

        Args:
            iteration_id: ID of the parent iteration.
            evaluations: List of evaluation dicts with parameter_values,
                objective_values, and optional suggestion_id.
        """
        async with self.session_factory() as session:
            for idx, eval_data in enumerate(evaluations):
                evaluation = EvaluationModel(
                    iteration_id=iteration_id,
                    batch_index=idx,
                    parameter_values_json=json.dumps(eval_data["parameter_values"]),
                    objective_values_json=json.dumps(eval_data["objective_values"]),
                    suggestion_id=eval_data.get("suggestion_id"),
                )
                session.add(evaluation)
            await session.commit()

    async def log_best_value(
        self,
        iteration_id: int,
        objective_name: str,
        best_value: float,
        best_params: dict[str, float],
        is_improvement: bool,
    ) -> None:
        """Log best value found up to this iteration.

        Args:
            iteration_id: ID of the parent iteration.
            objective_name: Name of the objective.
            best_value: Best value found so far.
            best_params: Parameters that achieved the best value.
            is_improvement: Whether this is an improvement over previous best.
        """
        async with self.session_factory() as session:
            bv = BestValueModel(
                iteration_id=iteration_id,
                objective_name=objective_name,
                best_value=best_value,
                best_params_json=json.dumps(best_params),
                is_improvement=is_improvement,
            )
            session.add(bv)
            await session.commit()

    async def get_run_info(self, run_id: str) -> dict[str, Any] | None:
        """Get basic run information.

        Args:
            run_id: ID of the run.

        Returns:
            Dictionary with run info, or None if not found.
        """
        async with self.session_factory() as session:
            result = await session.execute(select(RunModel).where(RunModel.id == run_id))
            run = result.scalar_one_or_none()
            if not run:
                return None

            return {
                "id": run.id,
                "name": run.name,
                "benchmark_function": run.benchmark_function,
                "n_parameters": run.n_parameters,
                "n_objectives": run.n_objectives,
                "batch_size": run.batch_size,
                "mcp_campaign_id": run.mcp_campaign_id,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "completed_at": run.completed_at.isoformat() if run.completed_at else None,
            }

    async def get_run_history(self, run_id: str) -> dict[str, Any]:
        """Get complete run history for visualization.

        Args:
            run_id: ID of the run.

        Returns:
            Dictionary with complete run history including iterations,
            method selections, evaluations, and best values.
        """
        async with self.session_factory() as session:
            # Load run with all related data
            result = await session.execute(
                select(RunModel)
                .where(RunModel.id == run_id)
                .options(
                    selectinload(RunModel.iterations).selectinload(IterationModel.method_selection),
                    selectinload(RunModel.iterations).selectinload(IterationModel.evaluations),
                    selectinload(RunModel.iterations).selectinload(IterationModel.model_metrics),
                    selectinload(RunModel.iterations).selectinload(IterationModel.best_value),
                )
            )
            run = result.scalar_one_or_none()
            if not run:
                return {}

            iterations_data = []
            for it in sorted(run.iterations, key=lambda x: x.iteration_number):
                it_data: dict[str, Any] = {
                    "iteration_number": it.iteration_number,
                    "timestamp": it.timestamp.isoformat() if it.timestamp else None,
                }

                # Method selection
                if it.method_selection:
                    ms = it.method_selection
                    it_data["method_selection"] = {
                        "model_type": ms.model_type,
                        "acquisition_function": ms.acquisition_function,
                        "optimization_strategy": ms.optimization_strategy,
                        "confidence": ms.confidence,
                        "explanation": ms.explanation,
                        "input_transforms": ms.get_input_transforms(),
                        "alternatives": ms.get_alternatives(),
                        "warnings": ms.get_warnings(),
                    }

                # Evaluations
                evaluations = []
                for ev in sorted(it.evaluations, key=lambda x: x.batch_index):
                    evaluations.append(
                        {
                            "batch_index": ev.batch_index,
                            "parameter_values": ev.get_parameter_values(),
                            "objective_values": ev.get_objective_values(),
                            "suggestion_id": ev.suggestion_id,
                        }
                    )
                it_data["evaluations"] = evaluations

                # Model metrics
                metrics = {}
                for m in it.model_metrics:
                    metrics[m.metric_name] = m.get_metric_data()
                it_data["model_metrics"] = metrics

                # Best value
                if it.best_value:
                    bv = it.best_value
                    it_data["best_value"] = {
                        "objective_name": bv.objective_name,
                        "value": bv.best_value,
                        "params": bv.get_best_params(),
                        "is_improvement": bv.is_improvement,
                    }

                iterations_data.append(it_data)

            return {
                "run": {
                    "id": run.id,
                    "name": run.name,
                    "benchmark_function": run.benchmark_function,
                    "n_parameters": run.n_parameters,
                    "n_objectives": run.n_objectives,
                    "batch_size": run.batch_size,
                    "mcp_campaign_id": run.mcp_campaign_id,
                    "created_at": run.created_at.isoformat() if run.created_at else None,
                    "completed_at": run.completed_at.isoformat() if run.completed_at else None,
                },
                "iterations": iterations_data,
            }

    async def get_method_selection_history(self, run_id: str) -> list[dict[str, Any]]:
        """Get method selections over all iterations.

        Args:
            run_id: ID of the run.

        Returns:
            List of method selection dicts ordered by iteration.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(IterationModel)
                .where(IterationModel.run_id == run_id)
                .options(selectinload(IterationModel.method_selection))
                .order_by(IterationModel.iteration_number)
            )
            iterations = result.scalars().all()

            history = []
            for it in iterations:
                if it.method_selection:
                    ms = it.method_selection
                    history.append(
                        {
                            "iteration": it.iteration_number,
                            "model_type": ms.model_type,
                            "acquisition_function": ms.acquisition_function,
                            "optimization_strategy": ms.optimization_strategy,
                            "confidence": ms.confidence,
                        }
                    )
            return history

    async def get_model_performance_history(self, run_id: str) -> list[dict[str, Any]]:
        """Get model performance metrics over all iterations.

        Args:
            run_id: ID of the run.

        Returns:
            List of metric dicts ordered by iteration.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(IterationModel)
                .where(IterationModel.run_id == run_id)
                .options(selectinload(IterationModel.model_metrics))
                .order_by(IterationModel.iteration_number)
            )
            iterations = result.scalars().all()

            history = []
            for it in iterations:
                metrics: dict[str, Any] = {"iteration": it.iteration_number}
                for m in it.model_metrics:
                    metrics[m.metric_name] = m.get_metric_data()
                history.append(metrics)
            return history

    async def get_best_value_progression(self, run_id: str) -> list[dict[str, Any]]:
        """Get best value progression over iterations.

        Args:
            run_id: ID of the run.

        Returns:
            List of best value dicts ordered by iteration.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(IterationModel)
                .where(IterationModel.run_id == run_id)
                .options(selectinload(IterationModel.best_value))
                .order_by(IterationModel.iteration_number)
            )
            iterations = result.scalars().all()

            progression = []
            for it in iterations:
                if it.best_value:
                    bv = it.best_value
                    progression.append(
                        {
                            "iteration": it.iteration_number,
                            "objective_name": bv.objective_name,
                            "best_value": bv.best_value,
                            "best_params": bv.get_best_params(),
                            "is_improvement": bv.is_improvement,
                        }
                    )
            return progression

    async def get_all_evaluations(self, run_id: str) -> list[dict[str, Any]]:
        """Get all evaluations from a run.

        Args:
            run_id: ID of the run.

        Returns:
            List of evaluation dicts ordered by iteration and batch index.
        """
        async with self.session_factory() as session:
            result = await session.execute(
                select(IterationModel)
                .where(IterationModel.run_id == run_id)
                .options(selectinload(IterationModel.evaluations))
                .order_by(IterationModel.iteration_number)
            )
            iterations = result.scalars().all()

            all_evals = []
            for it in iterations:
                for ev in sorted(it.evaluations, key=lambda x: x.batch_index):
                    all_evals.append(
                        {
                            "iteration": it.iteration_number,
                            "batch_index": ev.batch_index,
                            "parameter_values": ev.get_parameter_values(),
                            "objective_values": ev.get_objective_values(),
                        }
                    )
            return all_evals

    async def close(self) -> None:
        """Close database connections."""
        await self.engine.dispose()


if __name__ == "__main__":
    import asyncio

    async def test_database() -> None:
        """Test database operations."""
        db = AdaptiveBODatabase(Path("data/test_adaptive_bo.db"))
        await db.init_schema()

        # Create a test run
        run_id = await db.create_run(
            name="Test Run",
            benchmark="branin",
            n_params=2,
            n_objectives=1,
            batch_size=3,
            mcp_campaign_id="test-campaign-id",
        )
        print(f"Created run: {run_id}")

        # Log an iteration
        iteration_id = await db.log_iteration(run_id, 1)
        print(f"Created iteration: {iteration_id}")

        # Log method selection
        await db.log_method_selection(
            iteration_id,
            {
                "model_type": "SingleTaskGP",
                "acquisition_function": "qLogNEI",
                "optimization_strategy": "Sobol sequence",
                "confidence": "high",
                "explanation": "Standard GP for 2D optimization",
            },
        )
        print("Logged method selection")

        # Log evaluations
        await db.log_evaluations(
            iteration_id,
            [
                {"parameter_values": {"x1": 0.5, "x2": 0.5}, "objective_values": {"y": 10.0}},
                {"parameter_values": {"x1": 0.3, "x2": 0.7}, "objective_values": {"y": 5.0}},
            ],
        )
        print("Logged evaluations")

        # Log best value
        await db.log_best_value(
            iteration_id=iteration_id,
            objective_name="y",
            best_value=5.0,
            best_params={"x1": 0.3, "x2": 0.7},
            is_improvement=True,
        )
        print("Logged best value")

        # Retrieve history
        history = await db.get_run_history(run_id)
        print(f"\nRun history: {json.dumps(history, indent=2, default=str)}")

        await db.close()
        print("\nDatabase test complete!")

    asyncio.run(test_database())
