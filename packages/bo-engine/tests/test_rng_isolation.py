"""Tests for global ``torch`` RNG isolation in suggestion / sampling paths.

The BO-engine seeds ``torch``'s global RNG inside ``generate_next_batch``
(and inside both Thompson-sampling entry points) because BoTorch's
``optimize_acqf`` / ``MaxPosteriorSampling`` / ``posterior().rsample()``
all consult the process-wide generator rather than accepting an explicit
``torch.Generator``.  Without isolation, two concurrent calls — e.g. two
campaigns whose BO phase runs under ``asyncio.to_thread`` — would clobber
each other's seed and either diverge from their declared reproducibility
guarantee or race on each other's draws.

The fix wraps the mutation in ``torch.random.fork_rng(devices=[])``, which
saves the RNG state on entry and restores it on every return path.  These
tests assert that property directly so a future refactor that drops the
fork cannot re-introduce the bug silently.

References:
- PyTorch RNG fork semantics:
  https://pytorch.org/docs/stable/generated/torch.random.fork_rng.html
- BoTorch reproducibility guide:
  https://botorch.org/docs/reproducibility
"""

from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from bo_engine import (
    ObjectiveSpec,
    ObservationData,
    OptimizationSpec,
    ParameterSpec,
    ParameterType,
    create_and_fit_single_task_model,
    generate_next_batch,
)
from bo_engine.thompson_sampling import (
    ThompsonConfig,
    generate_thompson_samples,
)

RNG_PROBE_SIZE = 8
EXTERNAL_SEED = 7_919
INTERNAL_SEED = 31_337


@pytest.fixture
def single_obj_spec_seeded() -> OptimizationSpec:
    """Seeded two-parameter single-objective spec."""
    return OptimizationSpec(
        parameters=[
            ParameterSpec(name="x1", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
            ParameterSpec(name="x2", type=ParameterType.CONTINUOUS, bounds=(0.0, 1.0)),
        ],
        objectives=[ObjectiveSpec(name="y", minimize=True)],
        batch_size=1,
        random_seed=INTERNAL_SEED,
    )


@pytest.fixture
def observations_dense() -> list[ObservationData]:
    """Enough observations to trigger the BO (not initial-design) path."""
    return [
        ObservationData(parameter_values={"x1": 0.10, "x2": 0.20}, objective_values={"y": 5.00}),
        ObservationData(parameter_values={"x1": 0.50, "x2": 0.50}, objective_values={"y": 3.00}),
        ObservationData(parameter_values={"x1": 0.90, "x2": 0.10}, objective_values={"y": 4.00}),
        ObservationData(parameter_values={"x1": 0.30, "x2": 0.70}, objective_values={"y": 3.50}),
        ObservationData(parameter_values={"x1": 0.70, "x2": 0.40}, objective_values={"y": 4.50}),
    ]


def _rand_probe() -> torch.Tensor:
    """Draw a fixed-size sample from the global torch RNG."""
    return torch.rand(RNG_PROBE_SIZE)


class TestGenerateNextBatchRngIsolation:
    """``generate_next_batch`` must not mutate the global torch RNG."""

    def test_global_state_preserved_across_bo_call(
        self,
        single_obj_spec_seeded: OptimizationSpec,
        observations_dense: list[ObservationData],
    ) -> None:
        """Draws after the call match draws taken without the call.

        If the fork_rng wrapper is removed, ``torch.manual_seed`` inside
        ``generate_next_batch`` leaks into the caller's RNG stream and
        the post-call probe diverges from the baseline.
        """
        torch.manual_seed(EXTERNAL_SEED)
        baseline = _rand_probe()

        torch.manual_seed(EXTERNAL_SEED)
        generate_next_batch(
            single_obj_spec_seeded,
            observations_dense,
            batch_size=1,
            iteration=2,
        )
        after = _rand_probe()

        assert torch.equal(baseline, after), (
            "generate_next_batch leaked its internal seed into the caller's "
            "global torch RNG stream — fork_rng isolation appears to be broken."
        )

    def test_global_state_preserved_across_initial_design_call(
        self,
        single_obj_spec_seeded: OptimizationSpec,
    ) -> None:
        """Same invariant holds on the initial-design (early-return) branch."""
        torch.manual_seed(EXTERNAL_SEED)
        baseline = _rand_probe()

        torch.manual_seed(EXTERNAL_SEED)
        generate_next_batch(
            single_obj_spec_seeded,
            observations=[],
            batch_size=2,
            iteration=0,
        )
        after = _rand_probe()

        assert torch.equal(baseline, after), (
            "Initial-design branch of generate_next_batch leaked RNG state."
        )


class TestConcurrentBoCallsAreIsolated:
    """Two concurrent BO calls must not race on the global torch seed.

    The ``asyncio.to_thread`` offload introduced in §1.4 puts multiple
    BO invocations on Python's default thread pool, which shares the
    process-wide torch RNG.  Without ``fork_rng``, interleaved
    ``torch.manual_seed`` calls from two campaigns would overwrite
    each other and permanently shift the caller's RNG stream.

    We validate the property ``fork_rng`` *actually* guarantees —
    preservation of the outer RNG state across concurrent calls —
    rather than bitwise output determinism, which is confounded by
    multi-threaded BLAS non-determinism and is not the concern here.
    """

    def test_concurrent_calls_preserve_outer_rng_state(
        self,
        single_obj_spec_seeded: OptimizationSpec,
        observations_dense: list[ObservationData],
    ) -> None:
        """Outer RNG state survives two BO calls executed in a thread pool."""
        torch.manual_seed(EXTERNAL_SEED)
        baseline = _rand_probe()

        def _call(iteration: int) -> None:
            generate_next_batch(
                single_obj_spec_seeded,
                observations_dense,
                batch_size=1,
                iteration=iteration,
            )

        torch.manual_seed(EXTERNAL_SEED)
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(_call, (3, 11)))
        after = _rand_probe()

        assert torch.equal(baseline, after), (
            "Concurrent generate_next_batch calls under a thread pool "
            "leaked their internal seeds into the outer torch RNG stream."
        )


class TestThompsonSamplingRngIsolation:
    """``generate_thompson_samples`` must also preserve global RNG state."""

    def test_thompson_seed_does_not_leak(
        self,
        single_obj_spec_seeded: OptimizationSpec,
        observations_dense: list[ObservationData],
    ) -> None:
        """Seeded Thompson sampling leaves the caller's RNG stream intact."""
        train_x = torch.tensor(
            [[o.parameter_values["x1"], o.parameter_values["x2"]] for o in observations_dense],
            dtype=torch.float64,
        )
        train_y = torch.tensor(
            [[o.objective_values["y"]] for o in observations_dense],
            dtype=torch.float64,
        )
        bounds = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64)
        model = create_and_fit_single_task_model(train_x, train_y, bounds)

        torch.manual_seed(EXTERNAL_SEED)
        baseline = _rand_probe()

        torch.manual_seed(EXTERNAL_SEED)
        generate_thompson_samples(
            model=model,
            bounds=bounds,
            n_samples=2,
            config=ThompsonConfig(seed=INTERNAL_SEED),
            minimize=True,
        )
        after = _rand_probe()

        assert torch.equal(baseline, after), (
            "generate_thompson_samples leaked its internal seed into the "
            "caller's global torch RNG stream."
        )
