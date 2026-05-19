"""ID-hallucination recovery helpers.

Background: when an LLM client supplies a typo'd UUID, the standard
``CAMPAIGN_NOT_FOUND`` envelope's generic recovery_action forces the
agent to scan the full ``campaigns://list`` to recover. The
:mod:`bo_mcp_server.recovery` helpers compute a short list of fuzzy
matches so the agent can pick the right id on the next turn.

The similarity scoring is built on stdlib ``difflib.SequenceMatcher``
(Ratcliff/Obershelp). Reference docs:
https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.ratio.
"""

from __future__ import annotations

from uuid import uuid4

from bo_mcp_server.recovery import (
    MAX_SUGGESTIONS,
    SIMILARITY_THRESHOLD,
    find_similar_campaign_ids,
)


class TestFindSimilarCampaignIds:
    """Fuzzy-match suggestions for ``CAMPAIGN_NOT_FOUND`` envelopes."""

    def test_returns_empty_for_empty_inputs(self) -> None:
        assert find_similar_campaign_ids("", []) == []
        assert find_similar_campaign_ids("abc", []) == []
        assert find_similar_campaign_ids("", ["x"]) == []

    def test_single_hex_typo_is_surfaced(self) -> None:
        """One-character edit on a UUID should round-trip."""
        true_id = "12345678-1234-1234-1234-123456789abc"
        # Change the last hex digit; keep the rest identical.
        typo = "12345678-1234-1234-1234-123456789abd"
        suggestions = find_similar_campaign_ids(typo, [true_id])
        assert suggestions == [true_id]

    def test_unrelated_uuids_drop_below_threshold(self) -> None:
        """Random unrelated UUIDs must not appear as suggestions."""
        candidate = str(uuid4())
        unrelated = [str(uuid4()) for _ in range(5)]
        suggestions = find_similar_campaign_ids(candidate, unrelated)
        # All entries are unrelated to ``candidate``; none should clear
        # the similarity threshold.
        assert suggestions == []

    def test_caps_at_max_suggestions(self) -> None:
        """The result list is bounded by ``MAX_SUGGESTIONS``."""
        base = "12345678-1234-1234-1234-123456789ab"
        # Five close-match candidates differing only in the last hex
        # digit so every one of them clears the threshold.
        candidates: list[str] = [f"{base}{tail}" for tail in "0123456789"]
        candidate = f"{base}a"  # equally close to all of them
        suggestions = find_similar_campaign_ids(candidate, candidates)
        assert 1 <= len(suggestions) <= MAX_SUGGESTIONS

    def test_threshold_is_strict(self) -> None:
        """Pairs at exactly the threshold are accepted; below it are dropped."""
        candidate = "abcdef"
        # Identical: ratio = 1.0 >= threshold
        assert find_similar_campaign_ids(candidate, ["abcdef"]) == ["abcdef"]
        # Completely different: ratio = 0.0 < threshold
        assert find_similar_campaign_ids(candidate, ["zzzzzz"]) == []

    def test_dash_insensitive(self) -> None:
        """UUIDs with and without dashes match the same canonical form."""
        true_id = "12345678-1234-1234-1234-123456789abc"
        no_dashes = "12345678123412341234123456789abc"
        suggestions = find_similar_campaign_ids(no_dashes, [true_id])
        assert suggestions == [true_id]

    def test_higher_similarity_sorts_first(self) -> None:
        """Best match wins the top slot in the suggestion list."""
        best = "12345678-1234-1234-1234-123456789abc"
        worse = "12345678-1234-1234-1234-aaaaaaaaaaaa"  # weaker but may still clear
        candidate = "12345678-1234-1234-1234-123456789abd"  # one hex off ``best``
        # Only the strong match is expected to clear the threshold; if
        # both make it, ``best`` must sort first.
        suggestions = find_similar_campaign_ids(
            candidate, [worse, best], threshold=SIMILARITY_THRESHOLD
        )
        assert suggestions[0] == best
