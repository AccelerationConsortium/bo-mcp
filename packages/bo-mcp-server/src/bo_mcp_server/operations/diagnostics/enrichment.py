"""Server-side diagnostics enrichment: objective ranges, model info, interpretations."""

from typing import Any

from bo_mcp_server.domain import CampaignSpec, Result


def compute_objective_ranges(
    spec: CampaignSpec,
    results: list[Result],
) -> dict[str, dict[str, Any]]:
    """Compute min/max ranges for each objective."""
    if not results:
        return {}
    objective_ranges = {}
    for obj in spec.objectives:
        values = [r.objective_values[obj.name] for r in results]
        objective_ranges[obj.name] = {
            "min": min(values),
            "max": max(values),
            "direction": obj.direction,
        }
    return objective_ranges


def get_model_info(
    spec: CampaignSpec,
    is_single_objective: bool,
    method_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Get model information summary.

    ``method_info`` is the backend's own ``select_methods`` report; when
    present it is authoritative (issue #57: the static text below described
    BoTorch regardless of the campaign's backend). The static block is kept
    only as a legacy fallback for the BoTorch path when the backend report
    is unavailable.
    """
    if method_info:
        return {
            "backend": spec.backend,
            "type": method_info.get("model_type"),
            "acquisition_function": method_info.get("acquisition_function"),
            "batch_strategy": method_info.get("optimization_strategy"),
            "kernel": method_info.get("kernel"),
            "input_warping": spec.use_input_warping,
        }

    acquisition_fn = spec.acquisition_method.value
    if acquisition_fn == "auto":
        acquisition_fn = (
            "noisy_expected_improvement" if is_single_objective else "hypervolume_improvement"
        )

    acquisition_desc = {
        "noisy_expected_improvement": "Log Noisy Expected Improvement (qLogNEI)",
        "expected_improvement": "Log Expected Improvement (qLogEI)",
        "hypervolume_improvement": "Log Expected Hypervolume Improvement (qLogNEHVI)",
        "scalarized_multi_objective": "Parallel EGO with Chebyshev (qLogNParEGO)",
        "cost_weighted_ei": "Expected Improvement per Unit cost (EIpu)",
        "multi_fidelity_kg": "Multi-Fidelity Knowledge Gradient (qMFKG)",
    }.get(acquisition_fn, acquisition_fn)

    model_type = (
        "SingleTaskGP (Gaussian Process)"
        if is_single_objective
        else "ModelListGP (Multi-output Gaussian Process)"
    )

    return {
        "backend": spec.backend,
        "type": model_type,
        "acquisition_function": acquisition_desc,
        "batch_strategy": "Sequential greedy optimization",
        "kernel": "Matern 5/2 with automatic relevance determination",
        "input_warping": spec.use_input_warping,
    }


def interpret_lengthscales(lengthscales: dict[str, float]) -> str:
    """Provide agent-friendly interpretation of lengthscales."""
    if not lengthscales:
        return "No lengthscale information available."

    sorted_params = sorted(lengthscales.items(), key=lambda x: x[1])

    most_important = sorted_params[0][0] if sorted_params else "unknown"
    least_important = sorted_params[-1][0] if sorted_params else "unknown"

    if len(sorted_params) == 1:
        return f"The parameter '{most_important}' is the only tunable parameter."

    return (
        f"Based on lengthscales, '{most_important}' has the strongest influence on the "
        f"objective, while '{least_important}' has the weakest influence. "
        "Smaller lengthscales indicate more sensitivity to changes."
    )


def enrich_diagnostics(
    diagnostics: dict[str, Any],
    spec: CampaignSpec,
    results: list[Result],
    is_single_objective: bool,
    method_info: dict[str, Any] | None = None,
) -> None:
    """Add server-side enrichments to backend diagnostics."""
    diagnostics["objective_ranges"] = compute_objective_ranges(spec, results)
    model_info = get_model_info(spec, is_single_objective, method_info)

    if diagnostics.get("hyperparameters") is not None:
        hp = diagnostics["hyperparameters"]
        hp["interpretation"] = interpret_lengthscales(hp.get("lengthscales", {}))
        # A fitted surrogate is the ground truth for the kernel label.
        if hp.get("kernel_type"):
            model_info["kernel"] = hp["kernel_type"]

    diagnostics["model_info"] = model_info


def enrich_outlier_results(
    outlier_data: dict[str, Any] | None,
    results: list[Result],
) -> dict[str, Any] | None:
    """Add server-side result IDs and recommendation to backend outlier data."""
    if outlier_data is None:
        return None

    outlier_results = outlier_data.get("outlier_results", [])
    for info in outlier_results:
        idx = info.get("result_index", -1)
        if 0 <= idx < len(results):
            info["result_id"] = str(results[idx].id)

    count = outlier_data.get("count", 0)
    if count > 0:
        outlier_data["recommendation"] = (
            f"Detected {count} potential outlier(s). "
            "These results deviate significantly from model predictions. "
            "Consider verifying these measurements for errors."
        )
    else:
        outlier_data["recommendation"] = "No outliers detected. Results appear consistent."

    return outlier_data
