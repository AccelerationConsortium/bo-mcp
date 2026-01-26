#!/usr/bin/env python3
"""Plotly visualizations for Adaptive BO Demo.

This module provides interactive visualizations for tracking optimization
progress, model performance, and method selection history.

Usage:
    from adaptive_bo_plots import create_dashboard, save_plots
    from adaptive_bo_database import AdaptiveBODatabase

    db = AdaptiveBODatabase(Path("data/adaptive_bo_demo.db"))
    dashboard = await create_dashboard(db, run_id, optimum=0.397887)
    dashboard.write_html("dashboard.html")
"""

from pathlib import Path
from typing import Any

import plotly.graph_objects as go
from adaptive_bo_database import AdaptiveBODatabase
from plotly.subplots import make_subplots


async def plot_optimization_progress(
    db: AdaptiveBODatabase,
    run_id: str,
    optimum: float | None = None,
) -> go.Figure:
    """Plot best value found over iterations.

    Shows:
    - Best value progression (line)
    - Individual evaluations (scatter)
    - Optimum reference line (if known)

    Args:
        db: Database instance.
        run_id: ID of the run.
        optimum: Known global optimum value (optional).

    Returns:
        Plotly figure.
    """
    progression = await db.get_best_value_progression(run_id)
    all_evals = await db.get_all_evaluations(run_id)

    if not progression:
        return _empty_figure("No optimization data available")

    # Extract data
    iterations = [p["iteration"] for p in progression]
    best_values = [p["best_value"] for p in progression]
    improvements = [p["is_improvement"] for p in progression]

    # All individual evaluations
    eval_iterations = [e["iteration"] for e in all_evals]
    eval_values = [list(e["objective_values"].values())[0] for e in all_evals]

    fig = go.Figure()

    # Individual evaluations (scatter)
    fig.add_trace(
        go.Scatter(
            x=eval_iterations,
            y=eval_values,
            mode="markers",
            name="Evaluations",
            marker={"size": 6, "color": "lightblue", "opacity": 0.6},
            hovertemplate="Iteration %{x}<br>Value: %{y:.4f}<extra></extra>",
        )
    )

    # Best value line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=best_values,
            mode="lines+markers",
            name="Best Found",
            line={"color": "blue", "width": 2},
            marker={"size": 8},
            hovertemplate="Iteration %{x}<br>Best: %{y:.4f}<extra></extra>",
        )
    )

    # Highlight improvements
    improvement_iters = [it for it, imp in zip(iterations, improvements, strict=True) if imp]
    improvement_vals = [bv for bv, imp in zip(best_values, improvements, strict=True) if imp]
    if improvement_iters:
        fig.add_trace(
            go.Scatter(
                x=improvement_iters,
                y=improvement_vals,
                mode="markers",
                name="Improvements",
                marker={"size": 12, "color": "green", "symbol": "star"},
                hovertemplate="Iteration %{x}<br>New Best: %{y:.4f}<extra></extra>",
            )
        )

    # Optimum reference line
    if optimum is not None:
        fig.add_hline(
            y=optimum,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Optimum: {optimum:.4f}",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title="Optimization Progress",
        xaxis_title="Iteration",
        yaxis_title="Objective Value",
        legend={"yanchor": "top", "y": 0.99, "xanchor": "right", "x": 0.99},
        hovermode="closest",
    )

    return fig


async def plot_convergence_analysis(
    db: AdaptiveBODatabase,
    run_id: str,
    optimum: float,
) -> go.Figure:
    """Plot convergence metrics.

    Shows:
    - Simple regret: |f(x_best) - f(x*)|
    - Log regret

    Args:
        db: Database instance.
        run_id: ID of the run.
        optimum: Known global optimum value.

    Returns:
        Plotly figure.
    """
    progression = await db.get_best_value_progression(run_id)

    if not progression:
        return _empty_figure("No convergence data available")

    iterations = [p["iteration"] for p in progression]
    best_values = [p["best_value"] for p in progression]

    # Calculate regret
    regrets = [abs(bv - optimum) for bv in best_values]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Simple Regret", "Log Regret"),
    )

    # Simple regret
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=regrets,
            mode="lines+markers",
            name="Regret",
            line={"color": "purple", "width": 2},
        ),
        row=1,
        col=1,
    )

    # Log regret (add small epsilon to avoid log(0))
    log_regrets = [max(r, 1e-10) for r in regrets]
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=log_regrets,
            mode="lines+markers",
            name="Log Regret",
            line={"color": "orange", "width": 2},
        ),
        row=1,
        col=2,
    )
    fig.update_yaxes(type="log", row=1, col=2)

    fig.update_layout(
        title="Convergence Analysis",
        showlegend=False,
    )
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="|f(x_best) - f*|", row=1, col=1)
    fig.update_yaxes(title_text="Log |f(x_best) - f*|", row=1, col=2)

    return fig


async def plot_model_performance(
    db: AdaptiveBODatabase,
    run_id: str,
) -> go.Figure:
    """Plot model quality metrics over time.

    Shows metrics like:
    - Number of observations
    - Confidence score

    Args:
        db: Database instance.
        run_id: ID of the run.

    Returns:
        Plotly figure.
    """
    performance = await db.get_model_performance_history(run_id)

    if not performance:
        return _empty_figure("No model performance data available")

    iterations = [p["iteration"] for p in performance]

    # Collect available metrics
    metric_names = set()
    for p in performance:
        for key in p:
            if key != "iteration":
                metric_names.add(key)

    if not metric_names:
        return _empty_figure("No performance metrics recorded")

    n_metrics = len(metric_names)
    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=list(metric_names),
    )

    colors = ["blue", "green", "orange", "red", "purple"]
    for idx, metric_name in enumerate(sorted(metric_names)):
        values = [p.get(metric_name) for p in performance]
        # Filter out None values
        valid_iters = [it for it, v in zip(iterations, values, strict=True) if v is not None]
        valid_values = [v for v in values if v is not None]

        if valid_values:
            fig.add_trace(
                go.Scatter(
                    x=valid_iters,
                    y=valid_values,
                    mode="lines+markers",
                    name=metric_name,
                    line={"color": colors[idx % len(colors)], "width": 2},
                ),
                row=1,
                col=idx + 1,
            )
            fig.update_xaxes(title_text="Iteration", row=1, col=idx + 1)

    fig.update_layout(
        title="Model Performance Over Time",
        showlegend=False,
    )

    return fig


async def plot_method_selection_history(
    db: AdaptiveBODatabase,
    run_id: str,
) -> go.Figure:
    """Visualize which methods were selected over time.

    Shows a timeline of model types, acquisition functions, and strategies.

    Args:
        db: Database instance.
        run_id: ID of the run.

    Returns:
        Plotly figure.
    """
    history = await db.get_method_selection_history(run_id)

    if not history:
        return _empty_figure("No method selection data available")

    iterations = [h["iteration"] for h in history]
    models = [h["model_type"] for h in history]
    acquisitions = [h["acquisition_function"] for h in history]
    strategies = [h["optimization_strategy"] for h in history]
    confidences = [h.get("confidence", "unknown") for h in history]

    fig = make_subplots(
        rows=4,
        cols=1,
        subplot_titles=("Model Type", "Acquisition Function", "Strategy", "Confidence"),
        vertical_spacing=0.08,
    )

    # Create category mappings for consistent colors
    def create_category_trace(
        data: list[str], row: int, colormap: dict[str, str] | None = None
    ) -> None:
        unique_vals = sorted(set(data))
        if colormap is None:
            colors_list = [
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
            ]
            colormap = {v: colors_list[i % len(colors_list)] for i, v in enumerate(unique_vals)}

        for val in unique_vals:
            val_iters = [it for it, d in zip(iterations, data, strict=True) if d == val]
            val_y = [val] * len(val_iters)
            fig.add_trace(
                go.Scatter(
                    x=val_iters,
                    y=val_y,
                    mode="markers",
                    name=val,
                    marker={"size": 15, "color": colormap.get(val, "#888888"), "symbol": "square"},
                    showlegend=row == 1,  # Only show legend for first subplot
                    hovertemplate=f"{val}<br>Iteration %{{x}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    create_category_trace(models, 1)
    create_category_trace(acquisitions, 2)
    create_category_trace(strategies, 3)

    # Confidence with special color mapping
    confidence_colors = {
        "high": "#2ca02c",
        "medium": "#ff7f0e",
        "low": "#d62728",
        "unknown": "#888888",
    }
    create_category_trace(confidences, 4, confidence_colors)

    fig.update_layout(
        title="Method Selection History",
        height=600,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    for i in range(1, 5):
        fig.update_xaxes(title_text="Iteration" if i == 4 else "", row=i, col=1)

    return fig


async def plot_parameter_exploration(
    db: AdaptiveBODatabase,
    run_id: str,
) -> go.Figure:
    """Plot parameter space exploration.

    For 2D problems, shows a scatter plot of evaluated points.
    For higher dimensions, shows parallel coordinates.

    Args:
        db: Database instance.
        run_id: ID of the run.

    Returns:
        Plotly figure.
    """
    run_info = await db.get_run_info(run_id)
    all_evals = await db.get_all_evaluations(run_id)

    if not all_evals or not run_info:
        return _empty_figure("No evaluation data available")

    n_params = run_info["n_parameters"]

    if n_params == 2:
        # 2D scatter plot
        return await _plot_2d_exploration(all_evals)
    else:
        # Parallel coordinates for higher dimensions
        return _plot_parallel_coordinates(all_evals)


async def _plot_2d_exploration(all_evals: list[dict[str, Any]]) -> go.Figure:
    """Create 2D scatter plot of parameter exploration."""
    param_names = list(all_evals[0]["parameter_values"].keys())[:2]
    x_vals = [e["parameter_values"][param_names[0]] for e in all_evals]
    y_vals = [e["parameter_values"][param_names[1]] for e in all_evals]
    obj_vals = [list(e["objective_values"].values())[0] for e in all_evals]
    iterations = [e["iteration"] for e in all_evals]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            marker={
                "size": 10,
                "color": obj_vals,
                "colorscale": "Viridis_r",
                "showscale": True,
                "colorbar": {"title": "Objective"},
            },
            text=[f"Iter {it}" for it in iterations],
            hovertemplate=(
                f"{param_names[0]}: %{{x:.3f}}<br>"
                f"{param_names[1]}: %{{y:.3f}}<br>"
                "Objective: %{marker.color:.4f}<br>"
                "%{text}<extra></extra>"
            ),
        )
    )

    # Mark the best point
    best_idx = obj_vals.index(min(obj_vals))
    fig.add_trace(
        go.Scatter(
            x=[x_vals[best_idx]],
            y=[y_vals[best_idx]],
            mode="markers",
            marker={"size": 20, "color": "red", "symbol": "star"},
            name=f"Best ({obj_vals[best_idx]:.4f})",
            hovertemplate=(
                f"BEST<br>{param_names[0]}: %{{x:.3f}}<br>"
                f"{param_names[1]}: %{{y:.3f}}<br>"
                f"Objective: {obj_vals[best_idx]:.4f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title="Parameter Space Exploration",
        xaxis_title=param_names[0],
        yaxis_title=param_names[1],
    )

    return fig


def _plot_parallel_coordinates(all_evals: list[dict[str, Any]]) -> go.Figure:
    """Create parallel coordinates plot for high-dimensional exploration."""
    param_names = list(all_evals[0]["parameter_values"].keys())
    obj_vals = [list(e["objective_values"].values())[0] for e in all_evals]

    dimensions = []
    for pname in param_names:
        values = [e["parameter_values"][pname] for e in all_evals]
        dimensions.append({"label": pname, "values": values})

    # Add objective as the last dimension
    obj_name = list(all_evals[0]["objective_values"].keys())[0]
    dimensions.append({"label": obj_name, "values": obj_vals})

    fig = go.Figure(
        data=go.Parcoords(
            line={
                "color": obj_vals,
                "colorscale": "Viridis_r",
                "showscale": True,
            },
            dimensions=dimensions,
        )
    )

    fig.update_layout(title="Parameter Space Exploration (Parallel Coordinates)")

    return fig


async def create_dashboard(
    db: AdaptiveBODatabase,
    run_id: str,
    optimum: float | None = None,
) -> go.Figure:
    """Create comprehensive dashboard with all plots.

    Combines:
    - Optimization progress (top left)
    - Convergence analysis (top right, if optimum known)
    - Method selection history (bottom left)
    - Parameter exploration (bottom right)

    Args:
        db: Database instance.
        run_id: ID of the run.
        optimum: Known global optimum value (optional).

    Returns:
        Plotly figure with subplots.
    """
    run_info = await db.get_run_info(run_id)

    if not run_info:
        return _empty_figure(f"Run {run_id} not found")

    # Get data for plots
    progression = await db.get_best_value_progression(run_id)
    all_evals = await db.get_all_evaluations(run_id)
    method_history = await db.get_method_selection_history(run_id)

    if not progression:
        return _empty_figure("No optimization data available")

    # Create 2x2 subplot layout
    has_optimum = optimum is not None

    subplot_titles = [
        "Optimization Progress",
        "Convergence" if has_optimum else "Model Confidence",
        "Method Selection",
        "Parameter Space",
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=subplot_titles,
        vertical_spacing=0.12,
        horizontal_spacing=0.1,
    )

    # --- Plot 1: Optimization Progress ---
    iterations = [p["iteration"] for p in progression]
    best_values = [p["best_value"] for p in progression]
    # Note: improvements data extracted but not used in dashboard (used in detailed progress plot)

    eval_iterations = [e["iteration"] for e in all_evals]
    eval_values = [list(e["objective_values"].values())[0] for e in all_evals]

    # Individual evaluations
    fig.add_trace(
        go.Scatter(
            x=eval_iterations,
            y=eval_values,
            mode="markers",
            name="Evaluations",
            marker={"size": 5, "color": "lightblue", "opacity": 0.6},
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Best value line
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=best_values,
            mode="lines+markers",
            name="Best Found",
            line={"color": "blue", "width": 2},
            marker={"size": 6},
        ),
        row=1,
        col=1,
    )

    # Optimum line
    if optimum is not None:
        fig.add_hline(
            y=optimum,
            line_dash="dash",
            line_color="red",
            row=1,  # type: ignore[arg-type]
            col=1,  # type: ignore[arg-type]
        )

    # --- Plot 2: Convergence or Confidence ---
    if has_optimum and optimum is not None:
        regrets = [abs(bv - optimum) for bv in best_values]
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=regrets,
                mode="lines+markers",
                name="Regret",
                line={"color": "purple", "width": 2},
                showlegend=True,
            ),
            row=1,
            col=2,
        )
        fig.update_yaxes(type="log", row=1, col=2)
    else:
        # Show confidence over time
        confidences = [h.get("confidence", "unknown") for h in method_history]
        conf_map = {"high": 3, "medium": 2, "low": 1, "unknown": 0}
        conf_values = [conf_map.get(c, 0) for c in confidences]
        method_iters = [h["iteration"] for h in method_history]
        fig.add_trace(
            go.Scatter(
                x=method_iters,
                y=conf_values,
                mode="lines+markers",
                name="Confidence",
                line={"color": "green", "width": 2},
                showlegend=True,
            ),
            row=1,
            col=2,
        )

    # --- Plot 3: Method Selection (simplified) ---
    if method_history:
        method_iters = [h["iteration"] for h in method_history]
        strategies = [h["optimization_strategy"] for h in method_history]

        # Map strategies to numeric values for plotting
        unique_strategies = sorted(set(strategies))
        strategy_map = {s: i for i, s in enumerate(unique_strategies)}
        strategy_values = [strategy_map[s] for s in strategies]

        fig.add_trace(
            go.Scatter(
                x=method_iters,
                y=strategy_values,
                mode="markers+lines",
                name="Strategy",
                marker={"size": 10, "color": "orange"},
                line={"color": "orange", "width": 1},
                text=strategies,
                hovertemplate="Iter %{x}: %{text}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        # Add strategy labels to y-axis
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(unique_strategies))),
            ticktext=unique_strategies,
            row=2,
            col=1,
        )

    # --- Plot 4: Parameter Exploration ---
    n_params = run_info["n_parameters"]
    if n_params == 2 and all_evals:
        param_names = list(all_evals[0]["parameter_values"].keys())[:2]
        x_vals = [e["parameter_values"][param_names[0]] for e in all_evals]
        y_vals = [e["parameter_values"][param_names[1]] for e in all_evals]
        obj_vals = [list(e["objective_values"].values())[0] for e in all_evals]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                name="Samples",
                marker={
                    "size": 8,
                    "color": obj_vals,
                    "colorscale": "Viridis_r",
                    "showscale": True,
                    "colorbar": {"title": "Obj", "x": 1.02, "len": 0.4, "y": 0.2},
                },
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text=param_names[0], row=2, col=2)
        fig.update_yaxes(title_text=param_names[1], row=2, col=2)
    elif all_evals:
        # Show objective value distribution for high-D
        fig.add_trace(
            go.Histogram(
                x=eval_values,
                name="Objective Distribution",
                marker_color="teal",
            ),
            row=2,
            col=2,
        )
        fig.update_xaxes(title_text="Objective Value", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=2)

    # Layout
    fig.update_layout(
        title={
            "text": f"Adaptive BO Dashboard: {run_info['name']}",
            "y": 0.98,
            "x": 0.5,
            "xanchor": "center",
        },
        height=700,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.15, "xanchor": "center", "x": 0.5},
    )

    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_yaxes(title_text="Objective", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_xaxes(title_text="Iteration", row=2, col=1)

    return fig


async def save_plots(
    db: AdaptiveBODatabase,
    run_id: str,
    output_dir: Path,
    optimum: float | None = None,
) -> list[Path]:
    """Generate and save all plots as HTML files.

    Args:
        db: Database instance.
        run_id: ID of the run.
        output_dir: Directory to save plots.
        optimum: Known global optimum value (optional).

    Returns:
        List of saved file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    # Optimization progress
    progress_fig = await plot_optimization_progress(db, run_id, optimum)
    progress_path = output_dir / "optimization_progress.html"
    progress_fig.write_html(str(progress_path))
    saved_files.append(progress_path)

    # Convergence (if optimum known)
    if optimum is not None:
        conv_fig = await plot_convergence_analysis(db, run_id, optimum)
        conv_path = output_dir / "convergence_analysis.html"
        conv_fig.write_html(str(conv_path))
        saved_files.append(conv_path)

    # Method selection history
    method_fig = await plot_method_selection_history(db, run_id)
    method_path = output_dir / "method_selection.html"
    method_fig.write_html(str(method_path))
    saved_files.append(method_path)

    # Parameter exploration
    param_fig = await plot_parameter_exploration(db, run_id)
    param_path = output_dir / "parameter_exploration.html"
    param_fig.write_html(str(param_path))
    saved_files.append(param_path)

    # Model performance
    perf_fig = await plot_model_performance(db, run_id)
    perf_path = output_dir / "model_performance.html"
    perf_fig.write_html(str(perf_path))
    saved_files.append(perf_path)

    # Dashboard
    dash_fig = await create_dashboard(db, run_id, optimum)
    dash_path = output_dir / "dashboard.html"
    dash_fig.write_html(str(dash_path))
    saved_files.append(dash_path)

    return saved_files


def _empty_figure(message: str) -> go.Figure:
    """Create an empty figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 16},
    )
    fig.update_layout(
        xaxis={"visible": False},
        yaxis={"visible": False},
    )
    return fig


if __name__ == "__main__":
    import asyncio

    async def test_plots() -> None:
        """Test plot generation with sample data."""
        # This requires a database with data - use the database test first
        db = AdaptiveBODatabase(Path("data/test_adaptive_bo.db"))

        # Check if we have any runs
        import aiosqlite

        db_path = Path("data/test_adaptive_bo.db")
        if not db_path.exists():
            print("No test database found. Run adaptive_bo_database.py first.")
            return

        async with aiosqlite.connect(db_path) as conn:
            cursor = await conn.execute("SELECT id FROM runs LIMIT 1")
            row = await cursor.fetchone()
            if not row:
                print("No runs in database. Run adaptive_bo_database.py first.")
                return
            run_id = row[0]

        print(f"Generating plots for run: {run_id}")
        output_dir = Path("data/test_plots")
        saved = await save_plots(db, run_id, output_dir, optimum=0.397887)

        print(f"Saved {len(saved)} plots:")
        for p in saved:
            print(f"  - {p}")

        await db.close()

    asyncio.run(test_plots())
