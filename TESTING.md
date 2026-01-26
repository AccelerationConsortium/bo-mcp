# Testing Strategy

This document describes the testing strategy for the BO-MCP monorepo, including how to run tests locally, how CI pipelines are structured, and how to handle stochastic test failures.

## Quick Reference

```bash
# Run all fast tests (what CI runs on PRs)
uv run pytest -m "not slow and not nightly and not docker and not postgres" packages/

# Run tests for a specific package
cd packages/bo-engine && uv run pytest -m "not slow"

# Run slow tests (MCMC-based)
uv run pytest -m "slow" packages/bo-engine/

# Run full test suite including nightly tests
uv run pytest -m "not docker and not postgres" packages/

# Recalibrate tolerances for stochastic tests
uv run python scripts/calibrate_test_tolerances.py
```

## Test Categories (Markers)

Tests are organized using pytest markers to enable selective execution:

| Marker | Description | Run in CI | Typical Duration |
|--------|-------------|-----------|------------------|
| `smoke` | Fast critical path tests | PR checks | < 5s each |
| (default) | Standard unit/integration tests | PR checks | < 30s each |
| `slow` | Long-running tests (MCMC, CV) | Main branch only | 30s - 5min each |
| `nightly` | Statistical/stochastic tests | Nightly schedule | Varies |
| `tutorial` | BoTorch tutorial reproduction | All | Varies |
| `deterministic` | Tests requiring torch determinism | All | Varies |
| `postgres` | Tests requiring PostgreSQL | Manual | Varies |
| `docker` | Tests requiring Docker | Manual | Varies |

### Selecting Tests by Marker

```bash
# Run only smoke tests (fastest)
uv run pytest -m smoke packages/

# Run everything except slow and nightly
uv run pytest -m "not slow and not nightly" packages/

# Run tutorial reproduction tests
uv run pytest -m tutorial packages/bo-engine/

# Run nightly statistical tests
uv run pytest -m nightly packages/
```

## CI Pipeline Structure

The CI pipeline uses a tiered approach to balance speed and coverage:

### 1. Fast Tests (PR Checks)
- **Trigger:** Every push and pull request
- **Duration:** ~2-3 minutes per package
- **Command:** `pytest -m "not slow and not nightly and not docker and not postgres"`
- **Must pass:** Yes (blocks merge)

### 2. Slow Tests (Main Branch)
- **Trigger:** Push to main branch only
- **Duration:** ~5-10 minutes per package
- **Command:** `pytest -m "slow and not nightly"`
- **Must pass:** No (tracked, but uses `continue-on-error`)

### 3. Nightly Tests (Statistical)
- **Trigger:** Scheduled at 3:00 AM UTC
- **Duration:** ~15-30 minutes total
- **Command:** `pytest -m "not docker and not postgres"` (all tests)
- **Purpose:** Catch statistical regressions over multiple runs

## Handling Stochastic Tests

Bayesian Optimization involves inherently stochastic operations (MCMC sampling, acquisition optimization, etc.). Tests are designed to handle this in two ways:

### Invariant-Based Testing (Always Pass)

Tests that verify properties that always hold, regardless of randomness:

```python
def test_optimization_invariants(self):
    # These assertions never fail due to randomness
    assert pareto_y.shape[0] >= 2  # Always find Pareto points
    assert hypervolumes[-1] >= hypervolumes[0]  # Never decreases
    assert all(0 <= x <= 1 for x in suggestions)  # Bounds respected
```

### Statistical Testing (Nightly)

Tests that verify statistical properties over multiple runs:

```python
@pytest.mark.nightly
def test_optimization_finds_good_tradeoffs_statistical(self):
    results = [run_optimization(seed=i) for i in range(5)]
    assert np.mean(results) < threshold  # Mean should be good
    assert np.percentile(results, 90) < loose_threshold  # 90th percentile
```

### Calibrated Tolerances

Tolerance thresholds are empirically calibrated using Monte Carlo simulation:

```bash
# Generate calibrated tolerances
uv run python scripts/calibrate_test_tolerances.py --iterations 100

# Example output:
# pareto_max 99th percentile: 7.2
# Recommended CI tolerance: 8.0 (99th + 10% margin)
# Recommended nightly tolerance: 6.0 (95th percentile)
```

Calibrated values are stored in `packages/bo-engine/tests/conftest.py`:

```python
PARETO_MAX_TOLERANCE_CI = 8.0      # Must always pass
PARETO_MAX_TOLERANCE_NIGHTLY = 6.0  # Statistical threshold
```

## Test Speed Optimization

### SAASBO Tests

SAASBO uses MCMC sampling which is slow. Tests use tiered MCMC settings:

| Tier | Warmup | Samples | Use Case |
|------|--------|---------|----------|
| CI-Fast | 8 | 4 | Fast functional tests |
| Smoke | 32 | 16 | Basic validation |
| Full Tutorial | 256 | 128 | Nightly/manual only |

### Deterministic Mode

For tests requiring reproducibility, use the `deterministic_seed` fixture:

```python
def test_something_deterministic(self, deterministic_seed):
    # torch.use_deterministic_algorithms is enabled
    # All seeds (torch, numpy, random) are set
    result = some_operation()
    assert result == expected_exact_value
```

## Adding New Tests

### Fast Test (Default)

```python
def test_my_feature(self):
    """Basic functionality test."""
    result = my_function()
    assert result is not None
```

### Slow Test

```python
@pytest.mark.slow
def test_full_mcmc_convergence(self):
    """Full MCMC convergence (slow)."""
    # Uses full MCMC settings
    ...
```

### Statistical Test

```python
@pytest.mark.nightly
def test_statistical_property(self):
    """Statistical test (nightly only)."""
    results = [run_with_seed(i) for i in range(10)]
    assert np.mean(results) < threshold
```

## Debugging Flaky Tests

If a test fails intermittently:

1. **Check if it's marked correctly:**
   - Stochastic tests should use invariant assertions or be `@pytest.mark.nightly`
   - MCMC tests should use CI-fast settings or be `@pytest.mark.slow`

2. **Recalibrate tolerances:**
   ```bash
   uv run python scripts/calibrate_test_tolerances.py --iterations 200
   ```

3. **Use statistical assertions:**
   ```python
   # Instead of:
   assert result < 5.0  # Can fail randomly

   # Use:
   assert pareto_y.shape[0] >= 2  # Invariant (always true)
   # OR
   @pytest.mark.nightly
   def test_statistical():
       results = [run(seed=i) for i in range(5)]
       assert np.mean(results) < 4.0  # Statistical (mean is stable)
   ```

4. **Check seed scope:**
   ```python
   # Make sure seed is set at the right level
   torch.manual_seed(42)
   # ... all random operations should be within seed scope
   ```

## xfailed Tests

Some tests are marked as `xfail` (expected failure) for documented reasons:

- **Input validation tests (9):** Validation features not yet implemented
- **Response structure tests (4):** API response features on roadmap
- **Reproducibility tests (3):** Full reproducibility features planned

These are tracked in `pyproject.toml` with `xfail_strict = true` - if an xfailed test starts passing, CI will fail to notify that the feature was implemented.

## Timeouts

All tests have a 120-second timeout by default (configurable in `pyproject.toml`):

```toml
[tool.pytest.ini_options]
timeout = 120  # 2 minutes per test
```

For slow tests, the CI uses `--timeout=300` (5 minutes).
