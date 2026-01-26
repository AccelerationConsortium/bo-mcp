"""Tests for automatic method selection."""

from bo_engine import (
    AcquisitionMethod,
    MethodSelection,
    ObjectiveSpec,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    select_methods,
)


def make_spec(
    n_params: int = 3,
    n_objectives: int = 1,
    has_categorical: bool = False,
    acquisition_method: AcquisitionMethod = AcquisitionMethod.AUTO,
    use_input_warping: bool = False,
) -> OptimizationSpec:
    """Create a test optimization spec."""
    parameters = [
        ParameterSpec(name=f"x{i}", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0))
        for i in range(n_params)
    ]
    if has_categorical:
        parameters.append(
            ParameterSpec(
                name="cat",
                type=ParameterType.CATEGORICAL,
                categories=["a", "b", "c"],
            )
        )
    objectives = [ObjectiveSpec(name=f"y{i}", minimize=True) for i in range(n_objectives)]
    return OptimizationSpec(
        parameters=parameters,
        objectives=objectives,
        acquisition_method=acquisition_method,
        use_input_warping=use_input_warping,
    )


class TestSelectMethods:
    """Test select_methods function."""

    def test_returns_method_selection(self) -> None:
        """select_methods returns a MethodSelection dataclass."""
        spec = make_spec()
        result = select_methods(spec, n_observations=0)
        assert isinstance(result, MethodSelection)

    def test_single_objective_model_type(self) -> None:
        """Single objective uses SingleTaskGP."""
        spec = make_spec(n_objectives=1)
        result = select_methods(spec, n_observations=10)
        assert result.model_type == "SingleTaskGP"

    def test_multi_objective_model_type(self) -> None:
        """Multi-objective uses ModelListGP."""
        spec = make_spec(n_objectives=2)
        result = select_methods(spec, n_observations=10)
        assert result.model_type == "ModelListGP"

    def test_single_objective_acquisition_auto(self) -> None:
        """AUTO mode selects qLogNEI for single objective."""
        spec = make_spec(n_objectives=1, acquisition_method=AcquisitionMethod.AUTO)
        result = select_methods(spec, n_observations=10)
        assert result.acquisition_function == "qLogNEI"

    def test_multi_objective_acquisition_auto(self) -> None:
        """AUTO mode selects qLogNEHVI for multi-objective."""
        spec = make_spec(n_objectives=2, acquisition_method=AcquisitionMethod.AUTO)
        result = select_methods(spec, n_observations=10)
        assert result.acquisition_function == "qLogNEHVI"

    def test_explicit_acquisition_respected(self) -> None:
        """Explicit acquisition method is used."""
        spec = make_spec(acquisition_method=AcquisitionMethod.QLOGEI)
        result = select_methods(spec, n_observations=10)
        assert result.acquisition_function == "qLogEI"

    def test_zero_observations_uses_sobol(self) -> None:
        """Zero observations triggers Sobol initial design."""
        spec = make_spec()
        result = select_methods(spec, n_observations=0)
        assert "Sobol" in result.optimization_strategy

    def test_few_observations_uses_lbfgsb(self) -> None:
        """Non-zero observations uses L-BFGS-B."""
        spec = make_spec(n_params=3)
        result = select_methods(spec, n_observations=5)
        assert "L-BFGS-B" in result.optimization_strategy

    def test_high_dim_recommends_turbo(self) -> None:
        """High-dimensional single-objective recommends TuRBO."""
        spec = make_spec(n_params=25, n_objectives=1)
        result = select_methods(spec, n_observations=10)
        assert "TuRBO" in result.optimization_strategy

    def test_high_dim_multi_obj_warns(self) -> None:
        """High-dimensional multi-objective warns about TuRBO limitation."""
        spec = make_spec(n_params=25, n_objectives=2)
        result = select_methods(spec, n_observations=10)
        assert any("TuRBO" in w for w in result.warnings)
        assert "L-BFGS-B" in result.optimization_strategy

    def test_categorical_adds_one_hot_transform(self) -> None:
        """Categorical parameters add one-hot encoding transform."""
        spec = make_spec(has_categorical=True)
        result = select_methods(spec, n_observations=10)
        assert any("One-hot" in t for t in result.input_transforms)

    def test_warping_adds_transform(self) -> None:
        """Input warping adds Kumaraswamy transform."""
        spec = make_spec(use_input_warping=True)
        result = select_methods(spec, n_observations=10)
        assert any("Kumaraswamy" in t for t in result.input_transforms)

    def test_normalize_always_included(self) -> None:
        """Normalize transform is always included."""
        spec = make_spec()
        result = select_methods(spec, n_observations=10)
        assert any("Normalize" in t for t in result.input_transforms)


class TestConfidenceLevel:
    """Test confidence level determination."""

    def test_zero_observations_high_confidence(self) -> None:
        """Zero observations gives high confidence (deterministic design)."""
        spec = make_spec(n_params=3)
        result = select_methods(spec, n_observations=0)
        assert result.confidence == "high"

    def test_few_observations_medium_confidence(self) -> None:
        """Few observations gives medium confidence."""
        spec = make_spec(n_params=5)
        result = select_methods(spec, n_observations=3)  # Less than 2*5
        assert result.confidence == "medium"

    def test_many_observations_high_confidence(self) -> None:
        """Many observations gives high confidence."""
        spec = make_spec(n_params=3)
        result = select_methods(spec, n_observations=10)  # >= 2*3
        assert result.confidence == "high"

    def test_few_observations_adds_warning(self) -> None:
        """Few observations adds warning about model quality."""
        spec = make_spec(n_params=5)
        result = select_methods(spec, n_observations=3)
        assert any("observations" in w.lower() for w in result.warnings)


class TestAlternatives:
    """Test alternative suggestions."""

    def test_single_objective_suggests_qlogei(self) -> None:
        """Single objective suggests qLogEI as alternative."""
        spec = make_spec(n_objectives=1)
        result = select_methods(spec, n_observations=10)
        assert any(alt.get("acquisition") == "qLogEI" for alt in result.alternatives)

    def test_multi_objective_suggests_parego(self) -> None:
        """Multi-objective suggests qLogNParEGO as alternative."""
        spec = make_spec(n_objectives=2)
        result = select_methods(spec, n_observations=10)
        assert any(alt.get("acquisition") == "qLogNParEGO" for alt in result.alternatives)


class TestExplanation:
    """Test explanation generation."""

    def test_explanation_not_empty(self) -> None:
        """Explanation is generated."""
        spec = make_spec()
        result = select_methods(spec, n_observations=10)
        assert len(result.explanation) > 0

    def test_explanation_mentions_objective_count(self) -> None:
        """Explanation mentions number of objectives."""
        spec = make_spec(n_objectives=2)
        result = select_methods(spec, n_observations=10)
        assert "2 objectives" in result.explanation

    def test_explanation_mentions_model(self) -> None:
        """Explanation mentions model type."""
        spec = make_spec()
        result = select_methods(spec, n_observations=10)
        assert "Model" in result.explanation

    def test_explanation_for_zero_observations(self) -> None:
        """Explanation mentions Sobol for zero observations."""
        spec = make_spec()
        result = select_methods(spec, n_observations=0)
        assert "Sobol" in result.explanation or "initial" in result.explanation.lower()
