"""Tests for GP model selection and comparison.

Model selection is only meaningful if (a) each candidate is scored on its
*own* cross-validated fits, and (b) the likelihood-based criteria are
evaluated in the space the model was fitted in. Both contracts are pinned
here:

* Candidates must produce **distinct** CV metrics. The failure mode being
  guarded against is a CV layer that ignores the candidate configuration
  (or serves one candidate's cached metrics to all others), which turns
  "model comparison" into a stable sort by insertion order.
* The log marginal likelihood must be evaluated against the model's own
  (outcome-transformed) training targets. Evaluating a standardized model
  against raw targets produces values that are astronomically wrong for
  any y not already ~N(0, 1), corrupting the "lml" and "bic" criteria.

References:
    - Rasmussen & Williams "GPML" Ch. 5 (Bayesian model selection; the
      marginal likelihood automatically penalizes mismatched kernels —
      the basis for the matched-vs-rough ranking tests below)
    - Snoek et al., "Input Warping for Bayesian Optimization of
      Non-Stationary Functions", ICML 2014
"""

from __future__ import annotations

import math

import pytest
import torch
from gpytorch.mlls import ExactMarginalLogLikelihood

from bo_engine.benchmarks import branin, branin_bounds
from bo_engine.cross_validation import CVConfig, clear_cv_cache
from bo_engine.model_selection import (
    DEFAULT_CANDIDATES,
    KernelType,
    ModelCandidate,
    ModelConfiguration,
    ModelSelectionConfig,
    compare_models,
    get_model_selection_summary,
)

BOUNDS_1D = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

MATCHED_SMOOTH = ModelCandidate(
    name="Matched (Matern 5/2)",
    kernel=KernelType.MATERN_52,
    use_warping=False,
    configuration=ModelConfiguration.STANDARD,
    description="Smoothness matched to the data-generating function.",
)
MIS_SPECIFIED_ROUGH = ModelCandidate(
    name="Mis-specified (Matern 1/2)",
    kernel=KernelType.MATERN_12,
    use_warping=False,
    configuration=ModelConfiguration.STANDARD,
    description="Exponential kernel on a smooth function.",
)


def _heavily_warped_dataset(n: int = 30, seed: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """1-D non-stationary function: slow near 0, oscillating near 1."""
    torch.manual_seed(seed)
    train_x = torch.rand(n, 1, dtype=torch.float64)
    train_y = torch.sin(8 * torch.pi * train_x**4) + 0.02 * torch.randn(n, 1, dtype=torch.float64)
    return train_x, train_y


def _smooth_noisy_dataset(n: int = 40, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Smooth single-period sine with non-negligible observation noise."""
    torch.manual_seed(seed)
    train_x = torch.rand(n, 1, dtype=torch.float64)
    train_y = torch.sin(2 * torch.pi * train_x) + 0.15 * torch.randn(n, 1, dtype=torch.float64)
    return train_x, train_y


class TestCandidateMetricsAreDistinct:
    """Each candidate must be cross-validated on its own model configuration."""

    def test_candidates_receive_distinct_cv_metrics(self) -> None:
        """At least two candidates must produce different CV RMSEs.

        Different kernels and input transforms fit (and therefore
        generalize) differently on a heavily warped function, so identical
        metrics across all candidates can only mean the candidate
        configuration never reached the CV computation.
        """
        clear_cv_cache()
        train_x, train_y = _heavily_warped_dataset()

        config = ModelSelectionConfig(
            candidates=DEFAULT_CANDIDATES[:3],
            cv_config=CVConfig(k_folds=5),
        )
        result = compare_models(train_x, train_y, BOUNDS_1D, config)

        rmses = [r.cv_metrics.rmse for r in result.all_results]
        assert all(math.isfinite(v) for v in rmses), f"non-finite CV RMSEs: {rmses}"
        assert len({round(v, 9) for v in rmses}) >= 2, (
            f"All candidates returned identical CV RMSE ({rmses}); the CV "
            "layer is not evaluating the candidate configurations."
        )

    def test_ranks_are_a_permutation_and_summary_consistent(self) -> None:
        clear_cv_cache()
        train_x, train_y = _heavily_warped_dataset()

        config = ModelSelectionConfig(
            candidates=DEFAULT_CANDIDATES[:3],
            cv_config=CVConfig(k_folds=5),
        )
        result = compare_models(train_x, train_y, BOUNDS_1D, config)

        ranks = sorted(r.rank for r in result.all_results)
        assert ranks == list(range(1, len(result.all_results) + 1))
        assert result.best_candidate.name == result.all_results[0].candidate.name
        assert result.best_candidate.name in result.recommendation

        summary = get_model_selection_summary(result)
        assert summary["selected_model"] == result.best_candidate.name
        assert summary["n_models_compared"] == len(result.all_results)


WARPED_MATERN = ModelCandidate(
    name="Warped (Matern 5/2)",
    kernel=KernelType.MATERN_52,
    use_warping=True,
    configuration=ModelConfiguration.WARPED,
    description="Kumaraswamy input warping for non-stationary functions.",
)


def _branin_dataset(n: int = 30, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Branin samples on its native (non-unit) bounds ``x1∈[-5,10], x2∈[0,15]``."""
    torch.manual_seed(seed)
    bounds = branin_bounds().to(dtype=torch.float64)
    lo, hi = bounds[0], bounds[1]
    train_x = lo + (hi - lo) * torch.rand(n, 2, dtype=torch.float64)
    train_y = branin(train_x).unsqueeze(-1)
    return train_x, train_y


class TestWarpedCandidateOnNonUnitBounds:
    """The warped candidate must fit on raw (non-unit) parameter spaces.

    The Kumaraswamy warp is only defined on the unit cube, so warping must be
    preceded by normalization (the ``normalize → warp`` chain used by the
    production model factory, cf. ``tests/test_input_warping.py``). A bare
    ``Warp`` on Branin's ``[-5,10]×[0,15]`` domain fails to fit and is scored
    ``inf`` — making the warped GP impossible to select where it would help.
    """

    def test_warped_candidate_cv_rmse_is_finite_on_branin_bounds(self) -> None:
        clear_cv_cache()
        train_x, train_y = _branin_dataset()
        bounds = branin_bounds().to(dtype=torch.float64)

        config = ModelSelectionConfig(
            candidates=[WARPED_MATERN],
            cv_config=CVConfig(k_folds=5),
        )
        result = compare_models(train_x, train_y, bounds, config)

        warped = result.all_results[0]
        assert math.isfinite(warped.cv_metrics.rmse), (
            "Warped candidate scored a non-finite CV RMSE on non-unit bounds — "
            "the normalize→warp chain is not being applied."
        )
        assert warped.cv_metrics.method != "failed"

    def test_warped_candidate_competes_with_standard_on_branin_bounds(self) -> None:
        """Both candidates must produce finite, comparable metrics on raw bounds."""
        clear_cv_cache()
        train_x, train_y = _branin_dataset()
        bounds = branin_bounds().to(dtype=torch.float64)

        config = ModelSelectionConfig(
            candidates=[MATCHED_SMOOTH, WARPED_MATERN],
            cv_config=CVConfig(k_folds=5),
        )
        result = compare_models(train_x, train_y, bounds, config)

        rmses = {r.candidate.name: r.cv_metrics.rmse for r in result.all_results}
        assert all(math.isfinite(v) for v in rmses.values()), (
            f"Non-finite CV RMSE on Branin bounds: {rmses}"
        )


class TestMisSpecifiedCandidateRanksBelow:
    """The marginal likelihood must demote a smoothness-mis-specified kernel.

    GPML Ch. 5: the marginal likelihood embodies an automatic Occam's
    razor — on a smooth function observed with noise, an exponential
    (Matern 1/2) kernel pays a substantial evidence penalty relative to the
    matched Matern 5/2. Empirically the LML gap on this dataset is 0.2-7
    nats across seeds, making the ranking stable (RMSE alone is a much
    weaker discriminator because the fitted noise absorbs mean error).
    """

    def test_mis_specified_kernel_ranks_below_matched(self) -> None:
        clear_cv_cache()
        train_x, train_y = _smooth_noisy_dataset(seed=0)

        config = ModelSelectionConfig(
            candidates=[MIS_SPECIFIED_ROUGH, MATCHED_SMOOTH],
            cv_config=CVConfig(k_folds=5),
            selection_criterion="lml",
        )
        result = compare_models(train_x, train_y, BOUNDS_1D, config)

        by_name = {r.candidate.name: r for r in result.all_results}
        matched = by_name[MATCHED_SMOOTH.name]
        rough = by_name[MIS_SPECIFIED_ROUGH.name]

        assert matched.rank < rough.rank, (
            f"Matched kernel (LML={matched.log_marginal_likelihood:.2f}) must "
            f"outrank the mis-specified one (LML={rough.log_marginal_likelihood:.2f})."
        )
        assert matched.log_marginal_likelihood > rough.log_marginal_likelihood

    @pytest.mark.nightly
    def test_mis_specified_kernel_ranks_below_matched_across_seeds(self) -> None:
        """Statistical form of the ranking test (TESTING.md nightly pattern)."""
        wins = 0
        n_seeds = 5
        for seed in range(n_seeds):
            clear_cv_cache()
            train_x, train_y = _smooth_noisy_dataset(seed=seed)
            config = ModelSelectionConfig(
                candidates=[MIS_SPECIFIED_ROUGH, MATCHED_SMOOTH],
                cv_config=CVConfig(k_folds=5),
                selection_criterion="lml",
            )
            result = compare_models(train_x, train_y, BOUNDS_1D, config)
            by_name = {r.candidate.name: r.rank for r in result.all_results}
            wins += by_name[MATCHED_SMOOTH.name] < by_name[MIS_SPECIFIED_ROUGH.name]

        assert wins >= n_seeds - 1, (
            f"Matched kernel won only {wins}/{n_seeds} seeds by LML; the "
            "likelihood criterion has lost its discriminating power."
        )


class TestLogMarginalLikelihoodEvaluation:
    """LML must be computed on the model's own (transformed) targets."""

    def test_lml_pinned_to_mll_on_train_targets(self) -> None:
        """Regression pin: reported LML equals gpytorch's MLL on
        ``model.train_targets`` scaled to a total over observations.
        """
        clear_cv_cache()
        train_x, train_y = _smooth_noisy_dataset(n=24, seed=3)
        n = train_x.shape[0]

        config = ModelSelectionConfig(
            candidates=[MATCHED_SMOOTH],
            cv_config=CVConfig(k_folds=4),
            keep_all_models=True,
        )
        result = compare_models(train_x, train_y, BOUNDS_1D, config)
        comparison = result.all_results[0]
        model = comparison.model
        assert model is not None

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        model.train()
        expected = mll(model(*model.train_inputs), model.train_targets).item() * n  # ty: ignore[unresolved-attribute]
        model.eval()

        assert comparison.log_marginal_likelihood == pytest.approx(expected, rel=1e-9)

    def test_lml_sane_for_unstandardized_targets(self) -> None:
        """Fitting on y with mean ~100 must not blow up the reported LML.

        A standardized model evaluated against raw targets yields LML values
        on the order of -10^3..-10^6 here; evaluated in its own target space
        the per-observation log likelihood stays O(1), so the total must
        stay above -10·n.
        """
        clear_cv_cache()
        train_x, train_y = _smooth_noisy_dataset(n=24, seed=3)
        n = train_x.shape[0]
        train_y_shifted = 100.0 + 5.0 * train_y

        config = ModelSelectionConfig(
            candidates=[MATCHED_SMOOTH],
            cv_config=CVConfig(k_folds=4),
        )
        result = compare_models(train_x, train_y_shifted, BOUNDS_1D, config)
        lml = result.all_results[0].log_marginal_likelihood

        assert math.isfinite(lml)
        assert lml > -10 * n, (
            f"LML={lml:.1f} is implausibly low for {n} standardized "
            "observations — it was evaluated against raw targets."
        )

    def test_lml_invariant_to_affine_target_rescaling(self) -> None:
        """Standardize maps any affine transform of y to identical targets,
        so the reported LML must be (numerically) unchanged.
        """
        clear_cv_cache()
        train_x, train_y = _smooth_noisy_dataset(n=24, seed=3)

        config = ModelSelectionConfig(
            candidates=[MATCHED_SMOOTH],
            cv_config=CVConfig(k_folds=4),
        )
        lml_base = (
            compare_models(train_x, train_y, BOUNDS_1D, config)
            .all_results[0]
            .log_marginal_likelihood
        )
        clear_cv_cache()
        lml_affine = (
            compare_models(train_x, 100.0 + 5.0 * train_y, BOUNDS_1D, config)
            .all_results[0]
            .log_marginal_likelihood
        )

        assert lml_affine == pytest.approx(lml_base, rel=1e-4)

    def test_bic_consistent_with_reported_lml(self) -> None:
        """BIC must be -2·LML + k·ln(n) for the reported LML."""
        clear_cv_cache()
        train_x, train_y = _smooth_noisy_dataset(n=24, seed=3)
        n = train_x.shape[0]

        config = ModelSelectionConfig(
            candidates=[MATCHED_SMOOTH],
            cv_config=CVConfig(k_folds=4),
            keep_all_models=True,
        )
        comparison = compare_models(train_x, train_y, BOUNDS_1D, config).all_results[0]
        model = comparison.model
        assert model is not None

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        expected_bic = -2 * comparison.log_marginal_likelihood + n_params * math.log(n)

        assert comparison.bic == pytest.approx(expected_bic, rel=1e-9)
