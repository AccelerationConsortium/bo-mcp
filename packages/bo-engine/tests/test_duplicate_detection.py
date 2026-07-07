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

from bo_engine.result_validation import detect_duplicates


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
