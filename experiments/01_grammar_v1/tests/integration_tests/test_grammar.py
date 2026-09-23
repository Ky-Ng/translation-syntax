"""Integration tests for Grammar V1 (src/cfg/grammar.py, documented in GRAMMAR_V1.md).

How the ambiguity test works
----------------------------
A CFG is ambiguous iff some string has two distinct parse trees. We cannot test
all strings (relative clauses recurse unboundedly), so we test exhaustively up
to a length bound:

1. ``enumerate_trees`` generates EVERY derivation tree whose yield has at most
   MAX_LEN tokens (recursion is bounded by the length budget, so this is
   exhaustive, not sampled). Ambiguity means two distinct trees share a yield,
   so it also cannot pair a <=MAX_LEN string with a longer one — the bound is
   airtight for all strings up to MAX_LEN.
2. Independently, every enumerated string is fed to nltk's ChartParser, which
   finds ALL parses; we assert it finds exactly one. This cross-checks the
   enumeration against nltk's own parser on the exact CFG object we ship.

The tests run on the frame-level grammar (terminal categories N, V_trans, ...
as terminals). Because category vocabularies are disjoint (asserted below),
every word-level sentence maps to exactly one frame, so frame-level
(un)ambiguity transfers to the word level.
"""

import itertools

import pytest
from nltk.parse.chart import ChartParser

from cfg.grammar import (
    LEXICON,
    RULES,
    TERMINAL_CATEGORIES,
    enumerate_trees,
    generate_frames,
    hf_cfg,
    hi_cfg,
    hi_to_hf,
)

# Covers relative-clause depth 2 (e.g. N C N C N V_trans V_trans V_trans N is
# 9 tokens; adding adverbs reaches 12-13). Raising it grows coverage
# exponentially in runtime; 13 runs in a few seconds.
MAX_LEN = 13

# Regression constants for MAX_LEN=13; they change iff the rule set changes.
N_TREES = 908
N_HF_STRINGS = 254


# ---------------------------------------------------------------------------
# well-formedness of the paired rule table
# ---------------------------------------------------------------------------

def test_hf_rules_are_permutations_of_hi_rules():
    """Each HF RHS reorders its HI RHS; no rule repeats a symbol (repeats would
    make the HI -> HF child mapping in hi_to_hf ill-defined)."""
    for lhs, hi_rhs, hf_rhs in RULES:
        assert sorted(hi_rhs) == sorted(hf_rhs), f"{lhs}: {hf_rhs} is not a permutation of {hi_rhs}"
        assert len(set(hi_rhs)) == len(hi_rhs), f"{lhs}: duplicate symbol in RHS {hi_rhs}"


def test_all_symbols_defined_and_reachable():
    lhs_symbols = {lhs for lhs, _, _ in RULES}
    for lhs, hi_rhs, _ in RULES:
        for symbol in hi_rhs:
            assert symbol in lhs_symbols or symbol in TERMINAL_CATEGORIES, f"undefined symbol {symbol}"
    reachable, frontier = set(), {"S"}
    while frontier:
        symbol = frontier.pop()
        reachable.add(symbol)
        for lhs, hi_rhs, _ in RULES:
            if lhs == symbol:
                frontier.update(set(hi_rhs) - reachable)
    assert lhs_symbols | set(TERMINAL_CATEGORIES) <= reachable, "unreachable symbols"


def test_lexicon_categories_equal_size_and_disjoint():
    sizes = {category: len(LEXICON[category]) for category in ("N", "V_intrans", "V_trans", "Adv")}
    assert set(sizes.values()) == {26}, f"categories must have 26 words each: {sizes}"
    for category, words in LEXICON.items():
        assert len(set(words)) == len(words), f"duplicate word within {category}"
    for cat_a, cat_b in itertools.combinations(LEXICON, 2):
        overlap = set(LEXICON[cat_a]) & set(LEXICON[cat_b])
        assert not overlap, f"{cat_a} and {cat_b} share words {overlap}: breaks word->frame uniqueness"


# ---------------------------------------------------------------------------
# ambiguity
# ---------------------------------------------------------------------------

def test_hi_grammar_unambiguous():
    """No two distinct HI derivations share a yield, and nltk's ChartParser
    agrees that every derivable string up to MAX_LEN has exactly one parse."""
    trees = enumerate_trees("hi", MAX_LEN)
    assert len(trees) == N_TREES
    frames = {" ".join(frame) for frame, _ in trees}
    assert len(frames) == N_TREES, "two distinct HI trees share a yield"

    parser = ChartParser(hi_cfg())
    for frame in frames:
        parses = list(parser.parse(frame.split()))
        assert len(parses) == 1, f"HI frame {frame!r} has {len(parses)} parses"


def test_hf_ambiguity_is_exactly_the_gap_collision():
    """HF is ambiguous, but only because subject-gap and object-gap relative
    clauses linearize identically: removing the object-gap rules makes HF
    unambiguous, and the minimal collision behaves as documented."""
    trees = enumerate_trees("hf", MAX_LEN)
    assert len(trees) == N_TREES, "HI and HF must have the same derivations, reordered"
    assert len({" ".join(frame) for frame, _ in trees}) == N_HF_STRINGS

    # the documented minimal collision: "dog chases that cat swims"
    parses = list(ChartParser(hf_cfg()).parse("N V_trans C N V_intrans".split()))
    assert len(parses) == 2
    labels = [{position.label() for position in parse.subtrees()} for parse in parses]
    assert any("CP_subj_gap" in ls for ls in labels) and any("CP_obj_gap" in ls for ls in labels)

    # without object gaps, HF has no ambiguity at all
    obj_gap_symbols = {"CP_obj_gap", "S_obj_gap", "VP_obj_gap"}
    subj_only = [r for r in RULES if not ({r[0], *r[1]} & obj_gap_symbols)]
    subj_only_trees = enumerate_trees("hf", MAX_LEN, rules=subj_only)
    assert len(subj_only_trees) == len({" ".join(f) for f, _ in subj_only_trees})


# ---------------------------------------------------------------------------
# HI -> HF mirroring (the hierarchical reversal rule)
# ---------------------------------------------------------------------------

def test_hi_to_hf_is_a_bijection_onto_hf_derivations():
    """Mirroring every HI tree yields exactly the set of HF trees, once each."""
    hi_trees = [tree for _, tree in enumerate_trees("hi", MAX_LEN)]
    mirrored = {hi_to_hf(tree).pformat() for tree in hi_trees}
    hf_trees = {tree.pformat() for _, tree in enumerate_trees("hf", MAX_LEN)}
    assert mirrored == hf_trees


def test_hi_frames_translate_to_hf_frames():
    for frame, tree in enumerate_trees("hi", MAX_LEN):
        hf_frame = hi_to_hf(tree).leaves()
        assert sorted(hf_frame) == sorted(frame), "mirroring must only reorder tokens"
        assert frame[0] == "N", "HI sentences always start with the subject noun"
        # note: HF frames need not start with N — a center-embedded intransitive
        # RC puts its verb first ("swims that cat dog chases")


# (HI sentence, HF sentence, in_distribution) — in distribution means the
# linear reversal rule (keep word 1, reverse the rest) agrees with the
# hierarchical translation; subject-attached RCs (center embedding) and
# object gaps are OOD
GOLDEN_PAIRS = [
    ("cat swims", "cat swims", True),
    ("cat chases dog", "cat dog chases", True),
    ("cat chases quickly dog", "cat dog quickly chases", True),
    ("ant sees cat that swims", "ant swims that cat sees", True),
    ("ant sees cat that chases dog", "ant dog chases that cat sees", True),
    ("cat that swims chases dog", "swims that cat dog chases", False),  # center embedding, HF starts with V
    ("cat that chases dog swims", "dog chases that cat swims", False),  # center embedding
    ("cat that dog chases swims", "dog chases that cat swims", False),  # object gap: HF string collides above
    ("cat that dog chases quickly sings", "dog quickly chases that cat sings", False),
]


@pytest.mark.parametrize("hi_sentence,hf_sentence,in_distribution", GOLDEN_PAIRS)
def test_golden_translation_pairs(hi_sentence, hf_sentence, in_distribution):
    """Word-level end-to-end: parse HI with the lexicalized CFG, mirror, read off HF."""
    parses = list(ChartParser(hi_cfg(lexicalized=True)).parse(hi_sentence.split()))
    assert len(parses) == 1, f"{hi_sentence!r} has {len(parses)} parses"
    assert " ".join(hi_to_hf(parses[0]).leaves()) == hf_sentence


@pytest.mark.parametrize("hi_sentence,hf_sentence,in_distribution", GOLDEN_PAIRS)
def test_linear_vs_hierarchical_reversal(hi_sentence, hf_sentence, in_distribution):
    """The linear rule matches the hierarchical translation exactly on the
    in-distribution examples and fails on every OOD one."""
    words = hi_sentence.split()
    linear = " ".join(words[:1] + words[:0:-1])
    assert (linear == hf_sentence) == in_distribution
