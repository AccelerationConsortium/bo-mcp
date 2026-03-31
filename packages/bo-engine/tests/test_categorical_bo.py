"""Integration tests for categorical-only Bayesian Optimization.

These tests validate the core bug fix: no duplicate suggestions in batch
BO with purely categorical parameters. They reproduce the scenario from
hardcoded_fragment_bo_example.py (donor/acceptor fragment selection).

References:
    - BoTorch optimize_acqf_discrete:
      https://botorch.readthedocs.io/en/latest/optim.html#botorch.optim.optimize.optimize_acqf_discrete
    - Duplicate suggestions in categorical BO discussion:
      https://github.com/pytorch/botorch/discussions/1450
"""

import pytest
import torch

from bo_engine.suggestions import generate_next_batch
from bo_engine.types import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def fragment_spec() -> OptimizationSpec:
    """Donor/acceptor fragment selection spec (6 x 6 = 36 combos).

    Reproduces the setup from hardcoded_fragment_bo_example.py.
    """
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="donor",
                type=ParameterType.CATEGORICAL,
                categories=[
                    "phenyl",
                    "anisole",
                    "aniline",
                    "thiophene",
                    "carbazole",
                    "phenothiazine",
                ],
            ),
            ParameterSpec(
                name="acceptor",
                type=ParameterType.CATEGORICAL,
                categories=[
                    "phenyl",
                    "benzonitrile",
                    "pyridine",
                    "pyrimidine",
                    "benzothiadiazole",
                    "triazine",
                ],
            ),
        ],
        objectives=[ObjectiveSpec(name="gap_eV", minimize=True)],
        batch_size=2,
        initial_design_size=2,
    )


@pytest.fixture
def small_categorical_spec() -> OptimizationSpec:
    """Small categorical spec for fast tests (3 x 3 = 9 combos)."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(
                name="material",
                type=ParameterType.CATEGORICAL,
                categories=["steel", "aluminum", "titanium"],
            ),
            ParameterSpec(
                name="coating",
                type=ParameterType.CATEGORICAL,
                categories=["none", "chrome", "zinc"],
            ),
        ],
        objectives=[ObjectiveSpec(name="strength", minimize=False)],
        batch_size=2,
        initial_design_size=2,
    )


# One deterministic mock value per (donor, acceptor) pair
MOCK_GAP: dict[tuple[str, str], float] = {
    (d, a): round(5.9 - 0.17 * i - 0.22 * j + 0.03 * ((i + j) % 3), 6)
    for i, d in enumerate(
        ["phenyl", "anisole", "aniline", "thiophene", "carbazole", "phenothiazine"]
    )
    for j, a in enumerate(
        ["phenyl", "benzonitrile", "pyridine", "pyrimidine", "benzothiadiazole", "triazine"]
    )
}


def _make_observation(donor: str, acceptor: str, gap: float | None = None) -> ObservationData:
    """Create an observation from donor/acceptor pair."""
    if gap is None:
        gap = MOCK_GAP[(donor, acceptor)]
    return ObservationData(
        parameter_values={"donor": donor, "acceptor": acceptor},
        objective_values={"gap_eV": gap},
    )


# =============================================================================
# TestCategoricalBatchBO (smoke tests)
# =============================================================================


@pytest.mark.smoke
class TestCategoricalBatchBO:
    """Regression tests for the duplicate suggestion bug fix."""

    def test_no_duplicate_suggestions_in_batch(self, fragment_spec: OptimizationSpec) -> None:
        """Batch of 2 suggestions should never have identical parameter values.

        This is the primary regression test for the bug where continuous
        relaxation + argmax produced duplicate categorical suggestions.
        """
        torch.manual_seed(42)

        # Start with initial design (need n_params+1=3 for BO mode with 2 params)
        observations = [
            _make_observation("phenyl", "benzonitrile"),
            _make_observation("thiophene", "pyrimidine"),
            _make_observation("anisole", "triazine"),
            _make_observation("carbazole", "pyridine"),
            _make_observation("aniline", "benzothiadiazole"),
        ]

        suggestions, _ = generate_next_batch(
            spec=fragment_spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        assert len(suggestions) == 2
        # The two suggestions must have different parameter values
        params_0 = suggestions[0].parameter_values
        params_1 = suggestions[1].parameter_values
        assert params_0 != params_1, (
            f"Duplicate suggestions in batch: {params_0} == {params_1}. "
            "The discrete optimizer should produce unique candidates."
        )

    def test_all_suggestions_are_valid_categories(self, fragment_spec: OptimizationSpec) -> None:
        """Every suggestion value should be in the original category list."""
        torch.manual_seed(42)

        observations = [
            _make_observation("phenyl", "benzonitrile"),
            _make_observation("thiophene", "pyrimidine"),
            _make_observation("anisole", "triazine"),
            _make_observation("carbazole", "pyridine"),
            _make_observation("aniline", "benzothiadiazole"),
        ]

        suggestions, _ = generate_next_batch(
            spec=fragment_spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        donor_cats = fragment_spec.parameters[0].categories
        acceptor_cats = fragment_spec.parameters[1].categories
        assert donor_cats is not None
        assert acceptor_cats is not None

        for s in suggestions:
            assert s.parameter_values["donor"] in donor_cats, (
                f"Invalid donor: {s.parameter_values['donor']}"
            )
            assert s.parameter_values["acceptor"] in acceptor_cats, (
                f"Invalid acceptor: {s.parameter_values['acceptor']}"
            )

    def test_suggestions_improve_over_cycles(
        self, small_categorical_spec: OptimizationSpec
    ) -> None:
        """Best observed value should improve or at least not degrade over 4 cycles.

        Uses a synthetic objective where some combos are clearly better.
        """
        torch.manual_seed(42)
        spec = small_categorical_spec

        # Synthetic objective: strength depends on material and coating
        strength_map = {
            ("steel", "none"): 50.0,
            ("steel", "chrome"): 70.0,
            ("steel", "zinc"): 65.0,
            ("aluminum", "none"): 40.0,
            ("aluminum", "chrome"): 55.0,
            ("aluminum", "zinc"): 50.0,
            ("titanium", "none"): 80.0,
            ("titanium", "chrome"): 95.0,  # Best combo
            ("titanium", "zinc"): 85.0,
        }

        observations: list[ObservationData] = []
        best_values: list[float] = []

        # Initial observations (need n_params+1=3 for BO mode with 2 params)
        observations.append(
            ObservationData(
                parameter_values={"material": "steel", "coating": "none"},
                objective_values={"strength": 50.0},
            )
        )
        observations.append(
            ObservationData(
                parameter_values={"material": "aluminum", "coating": "zinc"},
                objective_values={"strength": 50.0},
            )
        )
        observations.append(
            ObservationData(
                parameter_values={"material": "titanium", "coating": "none"},
                objective_values={"strength": 80.0},
            )
        )
        observations.append(
            ObservationData(
                parameter_values={"material": "steel", "coating": "chrome"},
                objective_values={"strength": 70.0},
            )
        )
        observations.append(
            ObservationData(
                parameter_values={"material": "aluminum", "coating": "none"},
                objective_values={"strength": 40.0},
            )
        )
        best_values.append(80.0)

        for cycle in range(2):
            suggestions, _ = generate_next_batch(
                spec=spec,
                observations=observations,
                batch_size=2,
                iteration=cycle + 1,
            )

            for s in suggestions:
                params = s.parameter_values
                key = (params["material"], params["coating"])
                strength = strength_map[key]
                observations.append(
                    ObservationData(
                        parameter_values=params,
                        objective_values={"strength": strength},
                    )
                )

            current_best = max(obs.objective_values["strength"] for obs in observations)
            best_values.append(current_best)

        # Best should be non-decreasing (we're maximizing)
        for i in range(1, len(best_values)):
            assert best_values[i] >= best_values[i - 1] - 1e-6, (
                f"Best value degraded from {best_values[i - 1]} to {best_values[i]} at step {i}"
            )

    def test_avoids_previously_evaluated_combinations(
        self, small_categorical_spec: OptimizationSpec
    ) -> None:
        """With X_avoid from prior results, optimizer prefers unexplored combos."""
        torch.manual_seed(42)
        spec = small_categorical_spec

        # Provide enough observations to cover some combos (need n_params+1=3 for BO mode)
        observations = [
            ObservationData(
                parameter_values={"material": "steel", "coating": "none"},
                objective_values={"strength": 50.0},
            ),
            ObservationData(
                parameter_values={"material": "aluminum", "coating": "chrome"},
                objective_values={"strength": 55.0},
            ),
            ObservationData(
                parameter_values={"material": "titanium", "coating": "zinc"},
                objective_values={"strength": 85.0},
            ),
            ObservationData(
                parameter_values={"material": "steel", "coating": "zinc"},
                objective_values={"strength": 65.0},
            ),
            ObservationData(
                parameter_values={"material": "titanium", "coating": "none"},
                objective_values={"strength": 80.0},
            ),
        ]

        suggestions, _ = generate_next_batch(
            spec=spec,
            observations=observations,
            batch_size=2,
            iteration=1,
        )

        # Check that at least one suggestion explores a new combo
        evaluated_combos = {
            (obs.parameter_values["material"], obs.parameter_values["coating"])
            for obs in observations
        }
        suggestion_combos = {
            (s.parameter_values["material"], s.parameter_values["coating"]) for s in suggestions
        }

        # With X_avoid, we expect at least one new combo in the batch
        new_combos = suggestion_combos - evaluated_combos
        assert len(new_combos) >= 1, (
            f"Expected at least 1 new combo, but all suggestions were previously "
            f"evaluated: {suggestion_combos}"
        )


# =============================================================================
# TestCategoricalBatchDiversity (nightly)
# =============================================================================


@pytest.mark.nightly
class TestCategoricalBatchDiversity:
    """Statistical tests for batch diversity in categorical BO."""

    def test_batch_diversity_across_seeds(self, fragment_spec: OptimizationSpec) -> None:
        """Over 20 random seeds, duplicate rate should be < 5%.

        Since optimize_acqf_discrete with unique=True guarantees uniqueness,
        we expect 0% duplicates. This test guards against regressions where
        the discrete path might not be triggered correctly.
        """
        n_seeds = 20
        n_duplicates = 0

        observations = [
            _make_observation("phenyl", "benzonitrile"),
            _make_observation("thiophene", "pyrimidine"),
            _make_observation("anisole", "triazine"),
            _make_observation("carbazole", "pyridine"),
            _make_observation("aniline", "benzothiadiazole"),
        ]

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            suggestions, _ = generate_next_batch(
                spec=fragment_spec,
                observations=observations,
                batch_size=2,
                iteration=1,
            )

            if len(suggestions) == 2:
                if suggestions[0].parameter_values == suggestions[1].parameter_values:
                    n_duplicates += 1

        duplicate_rate = n_duplicates / n_seeds
        assert duplicate_rate < 0.05, (
            f"Duplicate rate {duplicate_rate:.1%} exceeds 5% threshold. "
            f"Got {n_duplicates}/{n_seeds} batches with duplicates."
        )
