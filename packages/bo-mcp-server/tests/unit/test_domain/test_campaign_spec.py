"""Tests for CampaignSpec domain model."""

import pytest
from pydantic import ValidationError

from bo_engine.types import TargetMode
from bo_mcp_server.domain import (
    CampaignSpec,
    Constraint,
    ConstraintType,
    InputParameter,
    Objective,
    OutcomeConstraint,
    ParameterType,
    TurboConfig,
)


class TestInputParameter:
    """Tests for InputParameter validation."""

    def test_continuous_param_requires_bounds(self):
        """Continuous parameter must have bounds."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="temp",
                type=ParameterType.CONTINUOUS,
            )

    def test_continuous_param_valid_bounds(self):
        """Continuous parameter bounds must be valid."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="temp",
                type=ParameterType.CONTINUOUS,
                bounds=(100.0, 20.0),  # Lower > upper  # ty: ignore[invalid-argument-type]
            )

    def test_continuous_param_success(self):
        """Valid continuous parameter."""
        param = InputParameter(
            name="temp",
            type=ParameterType.CONTINUOUS,
            bounds=(20.0, 100.0),  # ty: ignore[invalid-argument-type]
        )
        assert param.name == "temp"
        assert param.bounds is not None
        assert param.bounds.lower == 20.0
        assert param.bounds.upper == 100.0

    def test_discrete_param_requires_values_or_bounds(self):
        """Discrete parameter must have values or bounds."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="count",
                type=ParameterType.DISCRETE,
            )

    def test_discrete_param_with_bounds(self):
        """Discrete parameter with bounds."""
        param = InputParameter(
            name="count",
            type=ParameterType.DISCRETE,
            bounds=(1.0, 10.0),  # ty: ignore[invalid-argument-type]
        )
        assert param.name == "count"

    def test_categorical_param_requires_categories(self):
        """Categorical parameter must have categories."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
            )

    def test_categorical_param_requires_at_least_2(self):
        """Categorical parameter needs at least 2 categories."""
        with pytest.raises(ValidationError):
            InputParameter(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
                categories=("only_one",),
            )

    def test_categorical_param_success(self):
        """Valid categorical parameter."""
        param = InputParameter(
            name="catalyst",
            type=ParameterType.CATEGORICAL,
            categories=("Pt", "Pd", "Rh"),
        )
        assert param.name == "catalyst"
        assert len(param.categories) == 3  # ty: ignore[invalid-argument-type]

    def test_continuous_param_rejects_discrete_values(self):
        """A continuous parameter carrying a discrete grid is contradictory.

        Backends ignore structural fields outside the declared type
        (mirroring BayBE, where a NumericalContinuousParameter takes only
        bounds and a NumericalDiscreteParameter only values:
        https://emdgroup.github.io/baybe/stable/userguide/parameters.html),
        so accepting the grid would silently optimize over the full
        continuous range instead of the caller's intended values.
        """
        with pytest.raises(ValidationError, match="does not take values"):
            InputParameter(
                name="temp",
                type=ParameterType.CONTINUOUS,
                bounds=(20.0, 100.0),  # ty: ignore[invalid-argument-type]
                values=(30.0, 50.0),
            )

    def test_continuous_param_rejects_categories(self):
        """Category labels on a continuous parameter are rejected."""
        with pytest.raises(ValidationError, match="does not take categories"):
            InputParameter(
                name="temp",
                type=ParameterType.CONTINUOUS,
                bounds=(20.0, 100.0),  # ty: ignore[invalid-argument-type]
                categories=("low", "high"),
            )

    def test_discrete_param_rejects_categories(self):
        """Category labels on a discrete parameter are rejected."""
        with pytest.raises(ValidationError, match="does not take categories"):
            InputParameter(
                name="count",
                type=ParameterType.DISCRETE,
                values=(1.0, 2.0),
                categories=("low", "high"),
            )

    def test_categorical_param_rejects_numeric_fields(self):
        """Bounds / values on a categorical parameter are rejected."""
        with pytest.raises(ValidationError, match="does not take bounds"):
            InputParameter(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
                categories=("Pt", "Pd"),
                bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            )
        with pytest.raises(ValidationError, match="does not take values"):
            InputParameter(
                name="catalyst",
                type=ParameterType.CATEGORICAL,
                categories=("Pt", "Pd"),
                values=(1.0,),
            )


class TestConstraintShape:
    """Tests for :class:`Constraint` shape-by-type invariants.

    Linear constraints must carry one coefficient per parameter; sum
    constraints are unweighted and must not carry coefficients. Pre-fix,
    the engine quietly rewrote a ``LINEAR`` constraint without
    coefficients into an unweighted sum (which has different semantics),
    so a typo'd input ran a different optimization than the caller
    asked for.
    """

    def test_linear_requires_coefficients(self):
        """``type=linear`` without ``coefficients`` is rejected."""
        with pytest.raises(ValidationError) as exc:
            Constraint(
                type=ConstraintType.LINEAR,
                parameters=("a", "b"),
                value=1.0,
            )
        assert "Linear constraint requires coefficients" in str(exc.value)

    def test_linear_coefficient_count_must_match_parameters(self):
        """``coefficients`` must align one-to-one with ``parameters``."""
        with pytest.raises(ValidationError) as exc:
            Constraint(
                type=ConstraintType.LINEAR,
                parameters=("a", "b"),
                value=1.0,
                coefficients=(1.0,),  # one coefficient, two parameters
            )
        assert "one coefficient per parameter" in str(exc.value)

    def test_sum_constraint_rejects_coefficients(self):
        """Sum-* constraints are unweighted; coefficients are not accepted."""
        for sum_type in (
            ConstraintType.SUM_EQUALS,
            ConstraintType.SUM_LESS_THAN,
            ConstraintType.SUM_GREATER_THAN,
        ):
            with pytest.raises(ValidationError) as exc:
                Constraint(
                    type=sum_type,
                    parameters=("a", "b"),
                    value=1.0,
                    coefficients=(0.5, 0.5),
                )
            assert "does not accept coefficients" in str(exc.value)

    def test_valid_linear_constraint(self):
        """Well-shaped linear constraint passes validation."""
        c = Constraint(
            type=ConstraintType.LINEAR,
            parameters=("a", "b"),
            value=1.0,
            coefficients=(0.5, 0.5),
        )
        assert c.coefficients == (0.5, 0.5)

    def test_valid_sum_constraint(self):
        """Well-shaped sum constraint passes validation."""
        c = Constraint(
            type=ConstraintType.SUM_EQUALS,
            parameters=("a", "b"),
            value=1.0,
        )
        assert c.coefficients is None


class TestObjective:
    """Tests for Objective validation."""

    def test_valid_minimize_objective(self):
        """Valid minimization objective."""
        obj = Objective(
            name="cost",
            direction="minimize",
        )
        assert obj.is_minimize is True

    def test_valid_maximize_objective(self):
        """Valid maximization objective."""
        obj = Objective(
            name="yield",
            direction="maximize",
        )
        assert obj.is_minimize is False

    def test_invalid_direction(self):
        """Invalid direction raises error."""
        with pytest.raises(ValidationError):
            Objective(
                name="cost",
                direction="invalid",
            )

    def test_log_transform_valid_with_target_mode_minimize(self):
        """``log_transform`` gates on the resolved goal, not the spelling.

        The engine consumes the collapsed minimize flag (``is_minimize``),
        so the richer ``target_mode='minimize'`` declaration is as valid
        with ``log_transform`` as the legacy ``direction='minimize'``
        (BoTorch's Log outcome transform requires strictly positive
        targets, hence the minimize-only restriction:
        https://botorch.org/docs/tutorials/custom_botorch_model_in_ax/
        and ``botorch.models.transforms.outcome.Log``). Pins the field
        documentation, which names both spellings.
        """
        obj = Objective(name="cost", target_mode=TargetMode.MINIMIZE, log_transform=True)
        assert obj.is_minimize is True
        assert obj.log_transform is True


class TestCampaignSpec:
    """Tests for CampaignSpec validation."""

    def test_valid_campaign_spec(self, sample_campaign_spec: CampaignSpec):
        """Valid campaign spec is created successfully."""
        assert sample_campaign_spec.name == "Test Campaign"
        assert sample_campaign_spec.n_parameters == 2
        assert sample_campaign_spec.n_objectives == 2
        assert sample_campaign_spec.is_multi_objective is True

    def test_campaign_spec_requires_parameters(self):
        """Campaign spec requires at least one parameter."""
        with pytest.raises(ValidationError):
            CampaignSpec(
                name="Test",
                parameters=(),
                objectives=(Objective(name="cost", direction="minimize"),),
            )

    def test_campaign_spec_requires_objectives(self):
        """Campaign spec requires at least one objective."""
        with pytest.raises(ValidationError):
            CampaignSpec(
                name="Test",
                parameters=(
                    InputParameter(
                        name="temp",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 100.0),  # ty: ignore[invalid-argument-type]
                    ),
                ),
                objectives=(),
            )

    def test_campaign_spec_immutable(self, sample_campaign_spec: CampaignSpec):
        """Campaign spec is immutable."""
        with pytest.raises(ValidationError):
            sample_campaign_spec.name = "New Name"

    @pytest.mark.parametrize("field", ["max_iterations", "initial_design_size"])
    @pytest.mark.parametrize("value", [0, -5])
    def test_iteration_budget_fields_reject_non_positive(self, field: str, value: int):
        """``max_iterations`` / ``initial_design_size`` must be >= 1 when set.

        A zero or negative ``max_iterations`` creates a born-dead campaign
        whose every generate returns BUDGET_EXCEEDED; the constraint
        mirrors the ``ge=1`` already enforced on ``max_observations`` /
        ``batch_size``.
        """
        with pytest.raises(ValidationError):
            CampaignSpec.model_validate(
                {
                    "name": "Test",
                    "parameters": [
                        {"name": "temp", "type": "continuous", "bounds": [0.0, 100.0]},
                    ],
                    "objectives": [{"name": "cost", "direction": "minimize"}],
                    field: value,
                }
            )

    def test_constraint_references_valid_parameters(self):
        """Constraint must reference valid parameters."""
        with pytest.raises(ValidationError):
            CampaignSpec(
                name="Test",
                parameters=(
                    InputParameter(
                        name="temp",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 100.0),  # ty: ignore[invalid-argument-type]
                    ),
                ),
                objectives=(Objective(name="cost", direction="minimize"),),
                constraints=(
                    Constraint(
                        type=ConstraintType.SUM_EQUALS,
                        parameters=("unknown_param",),
                        value=1.0,
                    ),
                ),
            )

    def test_outcome_constraint_rejects_unknown_objective(self):
        """A typo'd ``objective_name`` on an outcome constraint fails at intake.

        Pre-fix, the engine silently disabled outcome constraints
        referencing a missing objective (``return None`` in
        ``_build_outcome_constraint_models``). That produced
        unconstrained suggestions the caller believed were constrained
        — the worst class of BO bug.
        """
        with pytest.raises(ValidationError) as exc:
            CampaignSpec(
                name="Test",
                parameters=(
                    InputParameter(
                        name="x",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                    ),
                ),
                objectives=(Objective(name="yield", direction="maximize"),),
                outcome_constraints=(
                    OutcomeConstraint(objective_name="ield", threshold=0.5),  # typo
                ),
            )
        assert "Outcome constraint references unknown objective" in str(exc.value)

    def test_outcome_constraint_accepts_declared_objective(self):
        """Outcome constraint that names a declared objective passes intake."""
        spec = CampaignSpec(
            name="Test",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="yield", direction="maximize"),),
            outcome_constraints=(OutcomeConstraint(objective_name="yield", threshold=0.5),),
        )
        assert spec.outcome_constraints[0].objective_name == "yield"

    def test_get_parameter(self, sample_campaign_spec: CampaignSpec):
        """get_parameter returns correct parameter or None."""
        param = sample_campaign_spec.get_parameter("temperature")
        assert param is not None
        assert param.name == "temperature"

        missing = sample_campaign_spec.get_parameter("nonexistent")
        assert missing is None


class TestValueObjectImmutability:
    """Domain value objects must be deeply immutable.

    ``ConfigDict(frozen=True)`` alone only prevents attribute reassignment
    on the model shell; without converting collection fields to tuples,
    callers can still ``param.categories.append(...)`` and silently
    corrupt shared instances. These tests pin both layers of the
    contract: the frozen shell rejects ``param.name = "x"`` and the
    tuple-typed collections raise on mutating method calls. The
    Bounds/Objective/Constraint check confirms simple value objects are
    hashable, which downstream caches rely on for dedup keys.

    Reference: Pydantic v2 ``ConfigDict(frozen=True)`` documentation;
    Hynek Schlawack, "Hashes are Hard" (general guidance on hashable
    value objects in Python).
    """

    def test_input_parameter_attributes_are_frozen(self):
        param = InputParameter(
            name="catalyst",
            type=ParameterType.CATEGORICAL,
            categories=("Pt", "Pd"),
        )
        with pytest.raises(ValidationError):
            param.name = "rebranded"  # type: ignore[misc]  # pyright: ignore[reportAttributeAccessIssue]

    def test_input_parameter_categories_are_tuple(self):
        """Categorical parameters cannot be mutated through the categories field."""
        param = InputParameter(
            name="catalyst",
            type=ParameterType.CATEGORICAL,
            categories=("Pt", "Pd"),
        )
        assert isinstance(param.categories, tuple)
        with pytest.raises(AttributeError):
            param.categories.append("Rh")  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]

    def test_input_parameter_values_are_tuple(self):
        """Discrete-value parameters cannot be mutated through the values field."""
        param = InputParameter(
            name="dose",
            type=ParameterType.DISCRETE,
            values=(1.0, 2.0, 3.0),
        )
        assert isinstance(param.values, tuple)
        with pytest.raises(AttributeError):
            param.values.append(4.0)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]

    def test_campaign_spec_parameters_are_tuple(self, sample_campaign_spec: CampaignSpec):
        """``spec.parameters.append(...)`` must not silently mutate a shared spec."""
        assert isinstance(sample_campaign_spec.parameters, tuple)
        with pytest.raises(AttributeError):
            sample_campaign_spec.parameters.append(  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]  # ty: ignore[unresolved-attribute]
                InputParameter(name="extra", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),  # ty: ignore[invalid-argument-type]
            )

    def test_constraint_collections_are_tuples(self):
        """Constraint references and coefficients are tuples."""
        constraint = Constraint(
            type=ConstraintType.LINEAR,
            parameters=("x", "y"),
            value=1.0,
            coefficients=(0.5, 0.5),
        )
        assert isinstance(constraint.parameters, tuple)
        assert isinstance(constraint.coefficients, tuple)

    def test_value_objects_are_hashable(self):
        """Bounds/Objective/Constraint must be hashable so caches can key on them."""
        from bo_mcp_server.domain.campaign_spec import Bounds

        bounds = Bounds(lower=0.0, upper=1.0)
        objective = Objective(name="y", direction="minimize")
        constraint = Constraint(
            type=ConstraintType.SUM_EQUALS,
            parameters=("x", "y"),
            value=1.0,
        )
        # ``hash`` must not raise. Equal instances must hash equal.
        assert hash(bounds) == hash(Bounds(lower=0.0, upper=1.0))
        assert hash(objective) == hash(Objective(name="y", direction="minimize"))
        assert hash(constraint) == hash(
            Constraint(type=ConstraintType.SUM_EQUALS, parameters=("x", "y"), value=1.0),
        )


class TestOptionMappingImmutability:
    """``parameter_options`` and ``backend_options`` must be deeply immutable.

    The earlier tuple conversion only froze the sequence fields; option
    mappings stayed as plain dicts and could be mutated through subscript
    assignment (``p.parameter_options["baybe"]["encoding"] = "x"``). The
    Pydantic validator now wraps option mappings in nested
    :class:`types.MappingProxyType` views and the value object overrides
    ``__hash__`` so instances with option payloads remain hashable.

    Reference: Python ``types.MappingProxyType`` documentation
    (https://docs.python.org/3/library/types.html#types.MappingProxyType).
    """

    def test_input_parameter_options_outer_is_read_only(self):
        """Assigning a new backend key on the outer option mapping fails."""
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"encoding": "ohe"}},
        )
        assert param.parameter_options is not None
        with pytest.raises(TypeError):
            param.parameter_options["botorch"] = {  # type: ignore[index]  # pyright: ignore[reportIndexIssue]  # ty: ignore[invalid-assignment]
                "k": "v",
            }

    def test_input_parameter_options_inner_is_read_only(self):
        """Mutating an inner option value via subscript fails."""
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"encoding": "ohe"}},
        )
        assert param.parameter_options is not None
        with pytest.raises(TypeError):
            param.parameter_options["baybe"]["encoding"] = "int"  # type: ignore[index]  # pyright: ignore[reportIndexIssue]  # ty: ignore[invalid-assignment]

    def test_input_parameter_options_deeply_nested_mapping_is_read_only(self):
        """A mapping nested below the first-level inner dict must also be frozen."""
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"nested": {"a": 1}}},
        )
        assert param.parameter_options is not None
        with pytest.raises(TypeError):
            param.parameter_options["baybe"]["nested"]["a"] = 2  # type: ignore[index]  # pyright: ignore[reportIndexIssue]

    def test_input_parameter_options_deeply_nested_list_is_read_only(self):
        """A list nested inside an inner option dict must also be frozen (tuple)."""
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"items": [1, 2, 3]}},
        )
        assert param.parameter_options is not None
        items = param.parameter_options["baybe"]["items"]
        # Lists are recursively converted to tuples so ``append`` is gone.
        assert isinstance(items, tuple)
        with pytest.raises(AttributeError):
            items.append(4)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]

    def test_input_parameter_hash_is_stable_through_attempted_deep_mutation(self):
        """Hash before and after attempted deep mutation must be identical.

        The frozen view raises on the mutating call, so the value object
        cannot change. This test pins both halves: mutation fails AND
        the hash projection is unaffected by the failed attempt.
        """
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"nested": {"a": 1}, "items": [1, 2]}},
        )
        original_hash = hash(param)
        assert param.parameter_options is not None
        with pytest.raises(TypeError):
            param.parameter_options["baybe"]["nested"]["a"] = 99  # type: ignore[index]  # pyright: ignore[reportIndexIssue]
        with pytest.raises(AttributeError):
            param.parameter_options["baybe"]["items"].append(3)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        assert hash(param) == original_hash

    def test_input_parameter_options_source_dict_cannot_mutate_frozen_view(self):
        """A held reference to the source dict must not bleed into the frozen view."""
        inner = {"encoding": "ohe"}
        source = {"baybe": inner}
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options=source,
        )
        # Mutate the original after construction; the model must not see it.
        inner["encoding"] = "int"
        source["botorch"] = {"foo": "bar"}
        assert param.parameter_options is not None
        assert param.parameter_options["baybe"]["encoding"] == "ohe"
        assert "botorch" not in param.parameter_options

    def test_input_parameter_with_options_is_hashable(self):
        """Hashing a parameter that carries options must not raise."""
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"encoding": "ohe", "active_values": ["A", "B"]}},
        )
        # ``hash`` must not raise. The hash is stable across equal instances.
        other = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"encoding": "ohe", "active_values": ["A", "B"]}},
        )
        assert hash(param) == hash(other)
        # Usable as a dict key.
        bucket: dict[InputParameter, int] = {param: 1}
        assert bucket[other] == 1

    def test_input_parameter_serializes_options_as_plain_dict(self):
        """JSON round-trip must produce a plain dict (not ``mappingproxy``)."""
        param = InputParameter(
            name="x",
            type=ParameterType.CONTINUOUS,
            bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
            parameter_options={"baybe": {"encoding": "ohe"}},
        )
        dumped = param.model_dump()
        assert dumped["parameter_options"] == {"baybe": {"encoding": "ohe"}}
        assert isinstance(dumped["parameter_options"], dict)
        assert isinstance(dumped["parameter_options"]["baybe"], dict)

    def test_campaign_spec_backend_options_inner_is_read_only(
        self,
        sample_continuous_param: InputParameter,
        sample_objective_minimize: Objective,
    ):
        """Mutating an inner backend-option value via subscript fails."""
        spec = CampaignSpec(
            name="opts",
            parameters=(sample_continuous_param,),
            objectives=(sample_objective_minimize,),
            backend_options={"botorch": {"acquisition_optimizer": "lbfgsb"}},
        )
        assert spec.backend_options is not None
        with pytest.raises(TypeError):
            spec.backend_options["botorch"]["acquisition_optimizer"] = "scipy"  # type: ignore[index]  # pyright: ignore[reportIndexIssue]  # ty: ignore[invalid-assignment]

    def test_campaign_spec_backend_options_deeply_nested_is_read_only(
        self,
        sample_continuous_param: InputParameter,
        sample_objective_minimize: Objective,
    ):
        """Deep nested dict / list inside ``backend_options`` is also frozen."""
        spec = CampaignSpec(
            name="opts",
            parameters=(sample_continuous_param,),
            objectives=(sample_objective_minimize,),
            backend_options={
                "botorch": {"nested": {"a": 1}, "items": [10, 20]},
            },
        )
        assert spec.backend_options is not None
        with pytest.raises(TypeError):
            spec.backend_options["botorch"]["nested"]["a"] = 99  # type: ignore[index]  # pyright: ignore[reportIndexIssue]
        items = spec.backend_options["botorch"]["items"]
        assert isinstance(items, tuple)
        with pytest.raises(AttributeError):
            items.append(30)  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]

    def test_campaign_spec_backend_options_serializes_back_to_lists(
        self,
        sample_continuous_param: InputParameter,
        sample_objective_minimize: Objective,
    ):
        """``model_dump`` thaws frozen tuples back to lists for JSON symmetry."""
        spec = CampaignSpec(
            name="opts",
            parameters=(sample_continuous_param,),
            objectives=(sample_objective_minimize,),
            backend_options={"botorch": {"items": [10, 20]}},
        )
        dumped = spec.model_dump()["backend_options"]
        assert dumped == {"botorch": {"items": [10, 20]}}
        assert isinstance(dumped["botorch"]["items"], list)

    def test_campaign_spec_with_backend_options_is_hashable(
        self,
        sample_continuous_param: InputParameter,
        sample_objective_minimize: Objective,
    ):
        """Hashing a spec that carries backend_options must not raise."""
        spec_a = CampaignSpec(
            name="opts",
            parameters=(sample_continuous_param,),
            objectives=(sample_objective_minimize,),
            backend_options={"botorch": {"acquisition_optimizer": "lbfgsb"}},
        )
        spec_b = CampaignSpec(
            name="opts",
            parameters=(sample_continuous_param,),
            objectives=(sample_objective_minimize,),
            backend_options={"botorch": {"acquisition_optimizer": "lbfgsb"}},
        )
        assert hash(spec_a) == hash(spec_b)


class TestConvergenceToleranceValidation:
    """``convergence_tolerance`` is single-objective only.

    The stopping helper consumes a running-best trajectory over the first
    objective, so multi-objective campaigns must reject the field at create
    time rather than silently misinterpreting it.
    """

    def test_single_objective_convergence_tolerance_accepted(self):
        """One objective + ``convergence_tolerance`` is valid."""
        spec = CampaignSpec(
            name="single",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(Objective(name="y", direction="minimize"),),
            convergence_tolerance=0.01,
        )
        assert spec.convergence_tolerance == pytest.approx(0.01)

    def test_multi_objective_convergence_tolerance_rejected(self):
        """Two objectives + ``convergence_tolerance`` raises validation error."""
        with pytest.raises(ValidationError, match="convergence_tolerance"):
            CampaignSpec(
                name="multi",
                parameters=(
                    InputParameter(
                        name="x",
                        type=ParameterType.CONTINUOUS,
                        bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                    ),
                ),
                objectives=(
                    Objective(name="y1", direction="minimize"),
                    Objective(name="y2", direction="minimize"),
                ),
                convergence_tolerance=0.01,
            )

    def test_multi_objective_without_tolerance_accepted(self):
        """Multi-objective campaigns without the field still validate."""
        spec = CampaignSpec(
            name="multi",
            parameters=(
                InputParameter(
                    name="x",
                    type=ParameterType.CONTINUOUS,
                    bounds=(0.0, 1.0),  # ty: ignore[invalid-argument-type]
                ),
            ),
            objectives=(
                Objective(name="y1", direction="minimize"),
                Objective(name="y2", direction="minimize"),
            ),
        )
        assert spec.convergence_tolerance is None


class TestTurboConfigValidation:
    """Pydantic-level rejection of nonsensical TuRBO tolerance values.

    The schema is the only place where REST / MCP clients can be stopped
    before garbage propagates into the engine. Each test pins one failure
    mode that the previous schema accepted silently.

    Reference: Eriksson et al., NeurIPS 2019, Algorithm 1. The trust-region
    operating band requires ``length_min < initial_length <= length_max``
    with all three strictly positive, and the success / failure tolerances
    are integers counting consecutive batches before adapting the region.
    """

    def test_paper_defaults_validate(self) -> None:
        """The published defaults round-trip without raising."""
        config = TurboConfig()
        assert config.initial_length == pytest.approx(0.8)
        assert config.length_min == pytest.approx(0.5**7)
        assert config.length_max == pytest.approx(1.6)

    def test_negative_initial_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TurboConfig(initial_length=-1.0)

    def test_zero_length_min_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TurboConfig(length_min=0.0)

    def test_zero_length_max_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TurboConfig(length_max=0.0)

    def test_inverted_band_rejected(self) -> None:
        """``length_min >= length_max`` collapses the expand/contract zone."""
        with pytest.raises(ValidationError, match="strictly less than"):
            TurboConfig(length_min=2.0, length_max=1.0)

    def test_initial_length_outside_band_rejected(self) -> None:
        """An initial length above ``length_max`` would clamp on the first expand."""
        with pytest.raises(ValidationError, match="initial_length"):
            TurboConfig(initial_length=2.0, length_min=0.01, length_max=1.6)

    def test_initial_length_below_min_rejected(self) -> None:
        """An initial length under ``length_min`` triggers restart on iteration 1."""
        with pytest.raises(ValidationError, match="initial_length"):
            TurboConfig(initial_length=0.001, length_min=0.01, length_max=1.6)

    def test_success_tolerance_below_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TurboConfig(success_tolerance=0)

    def test_failure_tolerance_below_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TurboConfig(failure_tolerance=0)

    def test_failure_tolerance_none_accepted(self) -> None:
        """``None`` means "derive at TurboState construction time"."""
        config = TurboConfig(failure_tolerance=None)
        assert config.failure_tolerance is None
