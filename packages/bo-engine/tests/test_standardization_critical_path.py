"""Tests for output standardization on the suggestion critical path.

The model factories already attach ``Standardize(m=1)``; previously nothing
re-asserted the invariant after fit and nothing protected against a near-zero
empirical stddev produced by constant / replicate training targets. These
tests pin:

* The factories' ``Standardize`` divisor is clamped to the configured floor
  when the raw stddev is below it (``floor_standardize_stdvs``).
* :func:`post_fit_verification` emits a human-readable warning when the
  invariant fails or the divisor was floored, naming the affected objective.
* :func:`generate_next_batch` propagates those warnings to every
  :class:`SuggestionResult` in the produced batch via ``model_warnings`` so
  the backend can lift them onto :attr:`SuggestionBatch.warnings`.

References:
    - Eriksson et al. 2019, "Scalable Global Optimization via Local
      Bayesian Optimization" (TuRBO §3): expand/contract tolerances assume
      unit-scale targets; verifying the invariant post-fit guards the
      assumption.
    - Balandat et al. 2020, "BoTorch", §4: outcome transforms make
      acquisition geometry invariant to objective rescaling — when the
      transform is misconfigured the invariant breaks silently.
"""

from __future__ import annotations

import torch

from bo_engine.constants import STANDARDIZATION_STD_FLOOR
from bo_engine.models import (
    create_and_fit_single_task_model,
    inspect_standardize_stdvs,
    post_fit_verification,
)
from bo_engine.suggestions import generate_next_batch
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

SEED = 7
N_DIMS = 2


def _make_constant_objective_observations() -> tuple[
    OptimizationSpec, list[ObservationData], torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Build a spec + observations whose objective is *near*-constant.

    BoTorch's ``Standardize(m=1)`` short-circuits a *literal* constant
    column by setting stdvs = 1.0 (so the divisor is never zero); we
    use a near-constant signal (jitter just above ``min_stdv=1e-8`` but
    below ``STANDARDIZATION_STD_FLOOR=1e-4``) so the inspection helper
    sees a recorded stddev in the warning band.
    """
    spec = OptimizationSpec(
        parameters=[
            ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
            for i in range(N_DIMS)
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=1,
        random_seed=SEED,
    )
    torch.manual_seed(SEED)
    n_obs = max(2 * N_DIMS + 1, 5)
    train_x = torch.rand(n_obs, N_DIMS, dtype=torch.float64)
    # Near-constant column: tiny per-row jitter that lands in the band
    # ``(BoTorch.Standardize.min_stdv=1e-8, STANDARDIZATION_STD_FLOOR=1e-4)``
    # so BoTorch records a real (small) stddev and our explicit floor
    # gets exercised.
    base = torch.full((n_obs, 1), 0.42, dtype=torch.float64)
    jitter = 1e-6 * torch.linspace(-1.0, 1.0, n_obs, dtype=torch.float64).unsqueeze(-1)
    train_y = base + jitter
    bounds = torch.stack(
        [torch.zeros(N_DIMS, dtype=torch.float64), torch.ones(N_DIMS, dtype=torch.float64)]
    )
    observations = [
        ObservationData(
            parameter_values={f"x{i}": float(train_x[k, i].item()) for i in range(N_DIMS)},
            objective_values={"y": float(train_y[k, 0].item())},
        )
        for k in range(n_obs)
    ]
    return spec, observations, train_x, train_y, bounds


class TestInspectStandardizeStdvs:
    """Inspection must report the raw stddev without mutating the model.

    BoTorch's ``Standardize`` writes the empirical stddev used to
    normalize ``train_Y`` at fit time. A previous version of this code
    clamped that tensor post-fit, which broke the forward / inverse
    transform round-trip: the GP was fit on targets standardized with
    the raw tiny stddev, but the posterior was un-transformed with the
    floored value, distorting predictions on the user-facing scale. The
    inspection helper now only reads the stddev so the round-trip is
    preserved; downstream code surfaces near-constant data via warnings.
    """

    def test_inspection_reports_raw_stddev_without_mutation(self) -> None:
        _, _, train_x, train_y, bounds = _make_constant_objective_observations()
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        transform = model.outcome_transform
        stdvs = transform.stdvs  # ty: ignore[unresolved-attribute]
        before = float(stdvs.detach().reshape(-1).min().item())  # ty: ignore[call-non-callable]

        raw = inspect_standardize_stdvs(model)

        # The recorded stddev sits below the configured floor (near-constant data).
        assert raw[0] < STANDARDIZATION_STD_FLOOR
        # And — critically — the stored tensor is not mutated by inspection.
        after = float(stdvs.detach().reshape(-1).min().item())  # ty: ignore[call-non-callable]
        assert after == before
        # Sanity: the returned min matches what we read off the model.
        assert raw[0] == before


class TestStandardizeRoundTripPreserved:
    """Forward/inverse transforms must agree on near-constant training data.

    Regression for the reviewer's High 2 finding: the previous version of
    this code clamped ``Standardize.stdvs`` post-fit, which left the
    inverse transform using a different divisor than the forward
    transform applied at fit time. The posterior at the training points
    must round-trip to the user-facing scale within tight tolerance even
    when the recorded stddev is small.
    """

    def test_posterior_at_training_points_matches_raw_targets(self) -> None:
        _, _, train_x, train_y, bounds = _make_constant_objective_observations()
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        # post_fit_verification used to mutate the standardization divisor;
        # invoking it now must NOT shift the posterior round-trip.
        post_fit_verification(model, objective_names=["y"])

        model.eval()
        with torch.no_grad():
            posterior_mean = model.posterior(train_x).mean.squeeze(-1)
        # Posterior mean at the training inputs should sit close to the
        # raw targets (a near-constant column around 0.42). The previous
        # in-place clamp inflated the inverse-Standardize divisor and
        # pushed the posterior far away from 0.42 on the raw scale.
        raw_mean = float(train_y.mean().item())
        posterior_drift = float((posterior_mean - raw_mean).abs().max().item())
        assert posterior_drift < 1e-3, (
            f"Posterior at training points drifted {posterior_drift:.3e} from the "
            f"raw mean {raw_mean:.6f}; the forward / inverse Standardize "
            "round-trip was broken (the historical bug post-fit clamped stdvs "
            "in place)."
        )


class TestPostFitVerification:
    """``post_fit_verification`` surfaces named warnings for the suggestion path."""

    def test_constant_objective_produces_named_warning(self) -> None:
        _, _, train_x, train_y, bounds = _make_constant_objective_observations()
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        warnings_list = post_fit_verification(model, objective_names=["y"])

        assert warnings_list, "expected at least one warning for constant data"
        joined = " ".join(warnings_list)
        assert "objective 'y'" in joined
        assert "near-constant data" in joined

    def test_well_scaled_objective_produces_no_warning(self) -> None:
        spec, observations, _, _, _ = _make_constant_objective_observations()
        # Replace constant targets with a smooth signal so the invariant holds.
        torch.manual_seed(SEED)
        train_x = torch.rand(len(observations), N_DIMS, dtype=torch.float64)
        signal = torch.sin(3.0 * train_x[:, 0]) + 0.5 * train_x[:, 1]
        bounds = torch.stack(
            [torch.zeros(N_DIMS, dtype=torch.float64), torch.ones(N_DIMS, dtype=torch.float64)]
        )
        model = create_and_fit_single_task_model(train_x, signal.unsqueeze(-1), bounds)

        warnings_list = post_fit_verification(model, objective_names=[spec.objectives[0].name])
        assert warnings_list == []


class TestSuggestionWarningsPropagation:
    """``generate_next_batch`` must stamp the warnings onto every SuggestionResult."""

    def test_constant_objective_propagates_warning_to_batch(self) -> None:
        spec, observations, *_ = _make_constant_objective_observations()
        suggestions, _ = generate_next_batch(spec, observations, batch_size=2, iteration=1)

        assert suggestions, "expected at least one suggestion"
        for sr in suggestions:
            assert sr.model_warnings, "warnings must be propagated to every suggestion"
            assert any("near-constant data" in w for w in sr.model_warnings)
            assert any("objective 'y'" in w for w in sr.model_warnings)
