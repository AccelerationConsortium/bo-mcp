"""ID-hallucination recovery helpers for ``CAMPAIGN_NOT_FOUND`` envelopes.

When an LLM client supplies a campaign id that does not resolve, the
error envelope already carries a ``recovery_action`` string ("Use
campaigns://list resource to verify campaign exists and get correct
ID."). For *typo'd* UUIDs — a single transposed hex digit, a swapped
nibble — the agent then has to scan the full list and re-pick. This
module computes a short list of fuzzy matches that the operation layer
attaches under ``error.details.suggestions`` so the agent can recover
on the next turn without paging through the catalogue.

The matching strategy is deliberately small and stable:

* Compare the input id (lower-case, dashes stripped) to the same form
  of each candidate.
* Score by ``difflib.SequenceMatcher`` ratio. Strings that round-trip
  through ``UUID()`` are pre-filtered against an absolute character
  threshold so unrelated random UUIDs do not appear as low-quality
  noise.
* Cap the result at :data:`MAX_SUGGESTIONS`. The cap exists to keep the
  error envelope small (a UUID is 36 bytes; three suggestions plus
  surrounding JSON is well under a kilobyte) and to avoid masking the
  recovery action with too much noise.

Reference for the similarity heuristic: Python ``difflib`` documents
``SequenceMatcher.ratio`` as Ratcliff/Obershelp pattern matching, which
is the same algorithm the standard library exposes via
``difflib.get_close_matches`` — preferred over Levenshtein here
because the threshold semantics line up with the recovery contract
(0.85 ≈ "1-2 edits over a UUID-length string"). See
https://docs.python.org/3/library/difflib.html#difflib.SequenceMatcher.
"""

from __future__ import annotations

from difflib import SequenceMatcher

# Maximum number of fuzzy-match candidates surfaced under
# ``error.details.suggestions``. Three is enough for the LLM to pick
# one on the next turn without dominating the envelope.
MAX_SUGGESTIONS = 3

# Minimum SequenceMatcher.ratio to consider an id a plausible typo.
# 0.85 corresponds to ~1-2 hex-digit edits across a 36-char UUID
# (32 hex digits + 4 dashes); below that the input is more likely a
# different campaign than a typo.
SIMILARITY_THRESHOLD = 0.85


def _normalize(value: str) -> str:
    """Lowercase + strip dashes so UUID dashing variations do not bias the score."""
    return value.replace("-", "").lower()


def find_similar_campaign_ids(
    candidate: str,
    known_ids: list[str],
    *,
    limit: int = MAX_SUGGESTIONS,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[str]:
    """Return up to ``limit`` known ids that look like typos of ``candidate``.

    Args:
        candidate: The id the caller supplied (may be malformed).
        known_ids: All ids the caller could plausibly have meant. The
            caller is responsible for scoping this list (e.g. to the
            authenticated user); this helper does not enforce ACLs.
        limit: Cap on returned suggestions.
        threshold: Minimum similarity ratio (Ratcliff/Obershelp) to
            include an id.

    Returns:
        Ids ordered by descending similarity. Exact-prefix matches sort
        ahead of low-quality ratios at the same threshold so the
        natural "I dropped the last hex digit" case appears first.
    """
    if not candidate or not known_ids:
        return []

    normalized_candidate = _normalize(candidate)
    scored: list[tuple[float, str]] = []
    for known in known_ids:
        normalized = _normalize(known)
        ratio = SequenceMatcher(None, normalized_candidate, normalized).ratio()
        if ratio >= threshold:
            scored.append((ratio, known))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [known for _, known in scored[: max(0, limit)]]
