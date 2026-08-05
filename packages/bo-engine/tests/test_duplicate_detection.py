"""Duplicate detection semantics for categorical parameters.

Rows that share every numeric value but differ in a categorical value are
*different experiments* — running the same temperatures in a different
solvent is the normal shape of a categorical DOE screen, not a duplicate
measurement. A near-duplicate warning must therefore never fire on a
categorical mismatch: the mismatch contributes no numeric distance, so
without an explicit exclusion the row would sit at distance 0.0 and pass
any positive tolerance.

References:
    - Montgomery, "Design and Analysis of Experiments" — factorial designs
      re-use identical numeric factor levels across every categorical
      level by construction.
"""

from __future__ import annotations

import torch

from bo_engine.result_validation import (
    detect_duplicates,
    detect_duplicates_batch,
    duplicate_row_mask,
)


class TestCategoricalMismatchIsNotDuplicate:
    def test_same_numerics_different_category_is_not_flagged(self) -> None:
        duplicates = detect_duplicates(
            new_params={"temperature": 25.0, "pressure": 1.0, "solvent": "etoh"},
            existing_params=[{"temperature": 25.0, "pressure": 1.0, "solvent": "meoh"}],
        )
        assert duplicates == [], (
            "Identical numerics with a different categorical value is a normal "
            "categorical-DOE sibling, not a near-duplicate."
        )

    def test_identical_row_including_category_is_exact_duplicate(self) -> None:
        duplicates = detect_duplicates(
            new_params={"temperature": 25.0, "solvent": "etoh"},
            existing_params=[{"temperature": 25.0, "solvent": "etoh"}],
        )
        assert len(duplicates) == 1
        assert duplicates[0].is_exact

    def test_near_numeric_same_category_is_still_flagged(self) -> None:
        duplicates = detect_duplicates(
            new_params={"temperature": 25.0, "solvent": "etoh"},
            existing_params=[{"temperature": 25.0 + 1e-9, "solvent": "etoh"}],
        )
        assert len(duplicates) == 1

    def test_mixed_rows_only_categorical_siblings_excluded(self) -> None:
        duplicates = detect_duplicates(
            new_params={"temperature": 25.0, "solvent": "etoh"},
            existing_params=[
                {"temperature": 25.0, "solvent": "meoh"},  # sibling — excluded
                {"temperature": 25.0, "solvent": "etoh"},  # true duplicate — index 1
            ],
        )
        assert [d.index for d in duplicates] == [1]


class TestDuplicateRowMask:
    """Vectorized membership mask shares batch duplicate semantics.

    ``duplicate_row_mask`` must agree with ``detect_duplicates_batch`` row
    membership while preserving the input dtype: float64 coordinates at large
    magnitude are representable to ~1e-10 relative precision, whereas a
    float32 cast (spacing ~0.06 at 1e6, IEEE 754 single precision) collapses
    genuinely distinct experiments onto the same value.
    """

    def test_large_magnitude_float64_rows_stay_distinct(self) -> None:
        """Points 0.01 apart at magnitude 1e6 are not duplicates.

        In float32 both coordinates round to the same representable value
        (spacing 0.0625 at 1e6), so a float32 implementation would report a
        distance of zero and a false duplicate.
        """
        existing = torch.tensor([[1_000_000.0]], dtype=torch.float64)
        new = torch.tensor([[1_000_000.01]], dtype=torch.float64)
        assert duplicate_row_mask(new, existing).tolist() == [False]

    def test_true_near_duplicate_at_large_magnitude_is_flagged(self) -> None:
        existing = torch.tensor([[1_000_000.0]], dtype=torch.float64)
        new = torch.tensor([[1_000_000.0 + 5e-7]], dtype=torch.float64)
        assert duplicate_row_mask(new, existing).tolist() == [True]

    def test_mask_matches_detect_duplicates_batch_membership(self) -> None:
        torch.manual_seed(3)
        existing = torch.rand(15, 3, dtype=torch.float64)
        new = torch.rand(20, 3, dtype=torch.float64)
        new[4] = existing[7]
        new[11] = existing[2] + 1e-8

        mask = duplicate_row_mask(new, existing)
        flagged_rows = {new_idx for new_idx, _, _ in detect_duplicates_batch(new, existing)}
        assert {i for i, hit in enumerate(mask.tolist()) if hit} == flagged_rows
        assert mask[4]
        assert mask[11]

    def test_empty_existing_yields_all_false(self) -> None:
        new = torch.rand(4, 2, dtype=torch.float64)
        existing = torch.empty(0, 2, dtype=torch.float64)
        mask = duplicate_row_mask(new, existing)
        assert mask.shape == (4,)
        assert not mask.any()
