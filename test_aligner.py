"""Deterministic unit tests for the fast aligner and transcript state."""

import time

from transcript import TranscriptState, align, tokenize


def merge(fin, uns, new):
    """Run align and return the merged full transcript as a string."""
    fin_t = tokenize(fin)
    uns_t = tokenize(uns)
    new_t = tokenize(new)
    new_uns, _conf = align(fin_t, uns_t, new_t)
    return " ".join(fin_t + new_uns)


def check(name, got, want):
    status = "PASS" if got == want else "FAIL"
    print(f"[{status}] {name}: got={got!r} want={want!r}")
    assert got == want, f"{name}: {got!r} != {want!r}"


def test_aligner():
    # Test 1 - exact duplicate
    check("exact duplicate", merge("", "hello everyone", "hello everyone"), "hello everyone")

    # Test 2 / spec sliding overlap
    check(
        "sliding overlap",
        merge("thank you very much", "", "very much for watching"),
        "thank you very much for watching",
    )

    # Test 3 - word drift -> replace unstable
    check(
        "word drift",
        merge("", "there seem to be a different hobby", "there seems to be someone who has a different hobby"),
        "there seems to be someone who has a different hobby",
    )

    # Near-duplicate hypothesis (minor drift) -> suppressed, not re-printed
    check(
        "near-duplicate suppressed",
        merge("", "I was looking forward to it.", "I was looking forward to that."),
        "I was looking forward to it.",
    )

    # Test 4 - repeated phrase
    check("repeated phrase", merge("", "a mini mascot", "a mini mascot"), "a mini mascot")

    # Test 5 - correction + continuation
    check(
        "correction + continuation",
        merge("", "Sleeping, minimmascot!", "a mini mascot, and this one is blue."),
        "a mini mascot, and this one is blue.",
    )

    # Section 6.A - exact overlap
    check(
        "exact overlap",
        merge("", "and here is the next one", "here is the next one and then"),
        "and here is the next one and then",
    )

    # Section 6.B - partial overlap (new is a fuller reinterpretation)
    check(
        "partial overlap (expansion)",
        merge("", "different hobby", "seems to be someone who has a different hobby, and here"),
        "seems to be someone who has a different hobby, and here",
    )

    # Section 6.E - reinterpreted speech
    check("reinterpreted", merge("", "Sleeping, minimmascot!", "a mini mascot!"), "a mini mascot!")

    # New is a subset of old -> do not regress
    check("no regression", merge("", "thank you very much", "thank you"), "thank you very much")

    # Continuation after finalized prefix
    check(
        "finalized + continuation",
        merge("thank you very much", "for watching", "thank you very much for watching today"),
        "thank you very much for watching today",
    )


def test_finalization():
    # Overflow cap finalizes the leading sentence once the tail exceeds the cap.
    st = TranscriptState(rollback_window=100.0, words_per_second=0.01)  # cap = 1 word
    st.apply_hypothesis("hello there. world.")
    _v, fin, uns, _ = st.snapshot()
    assert fin == "hello there.", fin
    assert uns == "world.", uns

    # Version control: a stale fix must be rejected.
    st2 = TranscriptState(rollback_window=10.0, words_per_second=3.0)
    v1, _, _, _ = st2.apply_hypothesis("one two three")
    v2, _, _, _ = st2.apply_hypothesis("one two three four")
    assert v2 > v1
    assert st2.apply_fix(v1, tokenize("STALE")) is False, "stale fix must be rejected"
    assert st2.apply_fix(v2, tokenize("one two three four")) is True, "current fix must be accepted"

    # Finalized text is never modified by later hypotheses.
    st3 = TranscriptState(rollback_window=100.0, words_per_second=0.01)
    st3.apply_hypothesis("hello everyone welcome")
    fin_before = st3.snapshot()[1]
    st3.apply_hypothesis("hello everyone welcome again")
    fin_after = st3.snapshot()[1]
    assert fin_after.startswith(fin_before), "finalized prefix must be immutable"

    print("finalization tests passed")


if __name__ == "__main__":
    test_aligner()
    test_finalization()
    print("\nAll aligner tests passed.")
