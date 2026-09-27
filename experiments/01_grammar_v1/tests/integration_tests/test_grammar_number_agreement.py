"""Integration tests for the number-agreement grammar (src/cfg/grammar_number_agreement.py).

Same methodology as test_grammar.py (exhaustive enumeration of every
derivation up to a length bound, cross-checked against nltk's ChartParser on
the shipped CFG object), plus the properties that define this grammar:

1. It is Grammar V1 with a number feature: erasing the feature from the rules
   gives V1's rule set, every tree projects onto a V1 tree, and every V1 frame
   has exactly 2^(number of nouns) number variants.
2. Agreement: in every enumerated tree each verb carries the number of the
   noun it agrees with, and at the word level the gold-standard sentences
   parse uniquely while their minimal agreement violations have no parse.
3. HF ambiguity: V1's subject-gap / object-gap string collision survives only
   when the nouns involved agree in number.
"""

import itertools
from collections import Counter

import pytest
from cfg import grammar as v1
from cfg.grammar_number_agreement import (
    LEXICON,
    NUMBERS,
    RULES,
    TERMINAL_CATEGORIES,
    agreement_dependencies,
    enumerate_trees,
    hf_cfg,
    hi_cfg,
    hi_to_hf,
    n_attractors,
    number_of,
    strip_number,
    strip_number_tree,
    terminal_labels,
)
from nltk.parse.chart import ChartParser

MAX_LEN = 13  # same bound as test_grammar.py: covers RC depth 2 with adverbs
PARSER_MAX_LEN = 11  # the ChartParser cross-check grows ~2x per token; 11 runs in a few seconds

# Regression constants for MAX_LEN=13; they change iff the rule set changes.
N_TREES = 14_644
N_HF_STRINGS = 7_566


def _parse(sentence, cfg):
    return list(ChartParser(cfg).parse(sentence.split()))


def _words(tree, triple):
    noun, verb, kind = triple
    leaves = tree.leaves()
    return (leaves[noun], leaves[verb], kind)


# ---------------------------------------------------------------------------
# the grammar is V1 plus a number feature
# ---------------------------------------------------------------------------


def test_rules_are_v1_with_number_features():
    """Erasing the number feature maps the rule set onto V1's, and every rule
    with a feature is internally consistent: its lhs number (if any) recurs on
    every numbered rhs symbol except a free object NP."""
    stripped = {(strip_number(lhs), tuple(map(strip_number, hi)), tuple(map(strip_number, hf))) for lhs, hi, hf in RULES}
    assert stripped == set(v1.RULES)
    for lhs, hi_rhs, hf_rhs in RULES:
        assert sorted(hi_rhs) == sorted(hf_rhs), f"{lhs}: {hf_rhs} is not a permutation of {hi_rhs}"
        assert len(set(hi_rhs)) == len(hi_rhs), f"{lhs}: duplicate symbol in RHS {hi_rhs}"
        numbers = {number_of(s) for s in (lhs, *hi_rhs)} - {None}
        if strip_number(lhs).startswith("VP_trans") and len(hi_rhs) == 2 and strip_number(hi_rhs[1]) == "NP":
            numbers.discard(number_of(hi_rhs[1]))  # the object's number is free
        assert len(numbers) <= 1, f"{lhs} -> {hi_rhs} mixes numbers"


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


def test_lexicon_is_v1_plus_plural_paradigms():
    """The singular columns are the V1 lexicon (so V1 is the all-singular
    fragment); each numbered category has 26 forms; all categories are
    pairwise disjoint (word -> frame stays unique)."""
    for category in ("N", "V_intrans", "V_trans"):
        assert LEXICON[f"{category}_sg"] == v1.LEXICON[category]
        assert len(LEXICON[f"{category}_pl"]) == 26
    assert LEXICON["Adv"] == v1.LEXICON["Adv"] and LEXICON["C"] == v1.LEXICON["C"]
    for category, words in LEXICON.items():
        assert len(set(words)) == len(words), f"duplicate word within {category}"
    for cat_a, cat_b in itertools.combinations(LEXICON, 2):
        overlap = set(LEXICON[cat_a]) & set(LEXICON[cat_b])
        assert not overlap, f"{cat_a} and {cat_b} share words {overlap}: breaks word->frame uniqueness"


def test_trees_project_onto_v1_with_number_multiplicity():
    """Erasing the feature from every enumerated tree gives a V1 tree, and each
    V1 tree is hit exactly 2^(number of nouns) times (nouns pick a number
    freely, verbs are then fixed by agreement)."""
    projected = Counter(strip_number_tree(tree).pformat() for _, tree in enumerate_trees("hi", MAX_LEN))
    v1_trees = {tree.pformat(): frame.count("N") for frame, tree in v1.enumerate_trees("hi", MAX_LEN)}
    assert set(projected) == set(v1_trees)
    assert all(projected[key] == 2**n_nouns for key, n_nouns in v1_trees.items())


# ---------------------------------------------------------------------------
# ambiguity
# ---------------------------------------------------------------------------


def test_hi_grammar_unambiguous():
    """No two distinct HI derivations share a yield (exhaustive to MAX_LEN), and
    nltk's ChartParser finds exactly one parse per frame up to PARSER_MAX_LEN."""
    trees = enumerate_trees("hi", MAX_LEN)
    assert len(trees) == N_TREES
    frames = {" ".join(frame) for frame, _ in trees}
    assert len(frames) == N_TREES, "two distinct HI trees share a yield"

    parser = ChartParser(hi_cfg())
    for frame in frames:
        if len(frame.split()) <= PARSER_MAX_LEN:
            assert len(list(parser.parse(frame.split()))) == 1, f"HI frame {frame!r} is ambiguous"


@pytest.mark.parametrize("a,b,c,d", list(itertools.product(NUMBERS, repeat=4)))
def test_hf_gap_collision_requires_number_agreement(a, b, c, d):
    """V1's minimal HF collision "N V_trans C N V_intrans" (subject gap
    "cat that chases dog swims" vs object gap "cat that dog chases swims"
    both linearize as "dog chases that cat swims") splits by number: the
    subject-gap reading needs the RC verb and matrix verb to agree with the
    head noun (b == c == d), the object-gap reading needs the RC verb to
    agree with the embedded subject (a == b) and the matrix verb with the
    head (c == d). Only the all-same assignments stay ambiguous."""
    frame = f"N_{a} V_trans_{b} C N_{c} V_intrans_{d}"
    parses = _parse(frame, hf_cfg())
    expected = int(b == c == d) + int(a == b and c == d)
    assert len(parses) == expected, frame
    labels = [{strip_number(t.label()) for t in parse.subtrees()} for parse in parses]
    assert ("CP_subj_gap" in set().union(*labels)) == (b == c == d)
    assert ("CP_obj_gap" in set().union(*labels)) == (a == b and c == d)


def test_hf_ambiguity_is_exactly_the_gap_collision():
    """HF has the same derivations as HI; its only ambiguity is the gap
    collision (removing object gaps makes HF unambiguous)."""
    trees = enumerate_trees("hf", MAX_LEN)
    assert len(trees) == N_TREES
    assert len({" ".join(frame) for frame, _ in trees}) == N_HF_STRINGS
    obj_gap_symbols = {s for rule in RULES for s in (rule[0], *rule[1]) if strip_number(s) in ("CP_obj_gap", "S_obj_gap", "VP_obj_gap")}
    subj_only = [r for r in RULES if not ({r[0], *r[1]} & obj_gap_symbols)]
    subj_only_trees = enumerate_trees("hf", MAX_LEN, rules=subj_only)
    assert len(subj_only_trees) == len({" ".join(f) for f, _ in subj_only_trees})


def test_hi_to_hf_is_a_bijection_onto_hf_derivations():
    hi_trees = [tree for _, tree in enumerate_trees("hi", MAX_LEN)]
    mirrored = {hi_to_hf(tree).pformat() for tree in hi_trees}
    hf_trees = {tree.pformat() for _, tree in enumerate_trees("hf", MAX_LEN)}
    assert mirrored == hf_trees


# ---------------------------------------------------------------------------
# agreement, frame level: every verb carries its subject's number
# ---------------------------------------------------------------------------


def test_every_verb_agrees_with_its_controller():
    """In every enumerated tree, agreement_dependencies pairs each verb with
    exactly one noun of the same number, in HI and in the mirrored HF order."""
    for frame, tree in enumerate_trees("hi", MAX_LEN):
        for order_tree in (tree, hi_to_hf(tree)):
            labels = terminal_labels(order_tree)
            assert labels == order_tree.leaves()
            dependencies = agreement_dependencies(order_tree)
            verbs = [i for i, label in enumerate(labels) if strip_number(label) in ("V_intrans", "V_trans")]
            assert [verb for _, verb, _ in dependencies] == verbs, frame
            for noun, verb, kind in dependencies:
                assert strip_number(labels[noun]) == "N"
                assert number_of(labels[noun]) == number_of(labels[verb]), (frame, kind)
            assert sum(kind == "matrix" for _, _, kind in dependencies) == 1


def test_attractors_only_in_center_embedding():
    """At depth 1 an attractor exists iff the RC is subject-attached and its
    noun (RC object or embedded subject) differs in number from the matrix
    subject; object-attached RCs and no-RC frames never have one."""
    for frame, tree in enumerate_trees("hi", MAX_LEN):
        rc_nodes = [t for t in tree.subtrees() if strip_number(t.label()) in ("CP_subj_gap", "CP_obj_gap")]
        if len(rc_nodes) > 1:
            continue
        in_subject = bool(rc_nodes) and any(t is rc_nodes[0] for t in tree[0].subtrees())
        matrix_number = number_of(frame[0])
        # only nouns before the matrix verb can intervene; a matrix object
        # follows its verb and never counts
        matrix_verb = next(verb for _, verb, kind in agreement_dependencies(tree) if kind == "matrix")
        intervening = [t for t in frame[1:matrix_verb] if strip_number(t) == "N" and number_of(t) != matrix_number]
        assert n_attractors(tree) == int(in_subject and bool(intervening)), frame


# ---------------------------------------------------------------------------
# agreement, word level: gold-standard sentences
# ---------------------------------------------------------------------------

# (HI sentence, HF sentence, in_distribution, dependencies as (noun, verb, kind))
GOLDEN_PAIRS = [
    # V1 sentences: the all-singular fragment translates exactly as in test_grammar.py
    ("cat swims", "cat swims", True, [("cat", "swims", "matrix")]),
    ("cat chases dog", "cat dog chases", True, [("cat", "chases", "matrix")]),
    ("cat that dog chases swims", "dog chases that cat swims", False, [("dog", "chases", "obj_gap"), ("cat", "swims", "matrix")]),
    # plural subjects, any object number
    ("cats swim", "cats swim", True, [("cats", "swim", "matrix")]),
    ("cats chase dog", "cats dog chase", True, [("cats", "chase", "matrix")]),
    ("cat chases dogs", "cat dogs chases", True, [("cat", "chases", "matrix")]),
    ("cats chase quickly dogs", "cats dogs quickly chase", True, [("cats", "chase", "matrix")]),
    # subject-gap RC: the RC verb agrees with the head noun, wherever the head is
    ("cats that pursue dog swim", "dog pursue that cats swim", False, [("cats", "pursue", "subj_gap"), ("cats", "swim", "matrix")]),
    ("cat that pursues dogs swims", "dogs pursues that cat swims", False, [("cat", "pursues", "subj_gap"), ("cat", "swims", "matrix")]),
    ("cats that swim sleep", "swim that cats sleep", False, [("cats", "swim", "subj_gap"), ("cats", "sleep", "matrix")]),
    ("cat chases dogs that swim", "cat swim that dogs chases", True, [("cat", "chases", "matrix"), ("dogs", "swim", "subj_gap")]),
    ("cats chase dog that swims quickly", "cats quickly swims that dog chase", True, [("cats", "chase", "matrix"), ("dog", "swims", "subj_gap")]),
    # object-gap RC: the RC verb agrees with the embedded subject, the matrix verb with the head
    ("cats that dog hugs tease bird", "dog hugs that cats bird tease", False, [("dog", "hugs", "obj_gap"), ("cats", "tease", "matrix")]),
    ("cat that dogs hug teases birds", "dogs hug that cat birds teases", False, [("dogs", "hug", "obj_gap"), ("cat", "teases", "matrix")]),
    ("cats that dogs hug quickly sleep", "dogs quickly hug that cats sleep", False, [("dogs", "hug", "obj_gap"), ("cats", "sleep", "matrix")]),
    ("cat chases dogs that bird fears", "cat bird fears that dogs chases", False, [("cat", "chases", "matrix"), ("bird", "fears", "obj_gap")]),
    ("cats chase dog that birds fear", "cats birds fear that dog chase", False, [("cats", "chase", "matrix"), ("birds", "fear", "obj_gap")]),
]  # fmt: skip


@pytest.mark.parametrize("hi_sentence,hf_sentence,in_distribution,dependencies", GOLDEN_PAIRS)
def test_golden_translation_pairs(hi_sentence, hf_sentence, in_distribution, dependencies):
    """Word-level end-to-end: parse HI with the lexicalized CFG (uniquely),
    mirror, read off HF, and recover the agreement dependencies."""
    parses = _parse(hi_sentence, hi_cfg(lexicalized=True))
    assert len(parses) == 1, f"{hi_sentence!r} has {len(parses)} parses"
    assert " ".join(hi_to_hf(parses[0]).leaves()) == hf_sentence
    assert [_words(parses[0], triple) for triple in agreement_dependencies(parses[0])] == dependencies
    words = hi_sentence.split()
    linear = " ".join(words[:1] + words[:0:-1])
    assert (linear == hf_sentence) == in_distribution


# (grammatical, ungrammatical): minimal pairs differing in exactly one
# inflection, each violating one of the three dependencies or exhibiting
# agreement attraction (the verb agreeing with the nearest noun instead)
AGREEMENT_VIOLATIONS = [
    # matrix
    ("cat swims", "cat swim"),
    ("cats swim", "cats swims"),
    ("cats chase dog", "cats chases dog"),
    ("cat chases dogs", "cat chase dogs"),
    # subject-gap RC verb must agree with the head noun
    ("cats that pursue dog swim", "cats that pursues dog swim"),
    ("cat chases dogs that swim", "cat chases dogs that swims"),
    ("cats that swim sleep", "cats that swims sleep"),
    # matrix verb across a subject-attached RC: attraction to the RC object
    ("cats that pursue dog swim", "cats that pursue dog swims"),
    ("cat that pursues dogs swims", "cat that pursues dogs swim"),
    # object-gap RC verb must agree with the embedded subject, not the head noun
    ("cats that dog hugs tease bird", "cats that dog hug tease bird"),
    ("cat that dogs hug teases birds", "cat that dogs hugs teases birds"),
    ("cat chases dogs that bird fears", "cat chases dogs that bird fear"),
    # matrix verb across an object-gap RC: attraction to the embedded subject
    ("cats that dog hugs tease bird", "cats that dog hugs teases bird"),
    ("cat that dogs hug teases birds", "cat that dogs hug tease birds"),
]


@pytest.mark.parametrize("grammatical,ungrammatical", AGREEMENT_VIOLATIONS)
def test_agreement_violations_have_no_parse(grammatical, ungrammatical):
    good, bad = grammatical.split(), ungrammatical.split()
    assert len(good) == len(bad) and sum(a != b for a, b in zip(good, bad)) == 1, "not a minimal pair"
    assert len(_parse(grammatical, hi_cfg(lexicalized=True))) == 1
    assert _parse(ungrammatical, hi_cfg(lexicalized=True)) == [], f"{ungrammatical!r} should be ungrammatical"


# In HF order the RC verb's inflection decides the gap type: V1's collision
# "dog chases that cat swims" (subject gap "cat that chases dog swims" or
# object gap "cat that dog chases swims") splits once the head noun and the
# RC's other noun differ in number. (HF sentence, HI sources): flipping the
# RC verb of a one-reading string gives the *other* reading, not an error.
HF_GAP_READINGS = [
    ("dog pursue that cats swim", ["cats that pursue dog swim"]),  # subject gap: "pursue" agrees with "cats"
    ("dog pursues that cats swim", ["cats that dog pursues swim"]),  # object gap: "pursues" agrees with "dog"
    ("dog hugs that cats bird tease", ["cats that dog hugs tease bird"]),
    ("dog hug that cats bird tease", ["cats that hug dog tease bird"]),
    ("cat bird fears that dogs chases", ["cat chases dogs that bird fears"]),
    ("cat bird fear that dogs chases", ["cat chases dogs that fear bird"]),
    ("dog pursues that cat swims", ["cat that pursues dog swims", "cat that dog pursues swims"]),  # all singular: V1's collision
    ("dogs pursue that cats swim", ["cats that pursue dogs swim", "cats that dogs pursue swim"]),
    ("dogs pursue that cats swims", []),  # "swims" agrees with neither reading's subject
]


@pytest.mark.parametrize("hf_sentence,hi_sources", HF_GAP_READINGS)
def test_hf_rc_verb_number_selects_the_gap_reading(hf_sentence, hi_sources):
    """The HF string has exactly one parse per listed HI source, and each
    source translates to it; with no source it is ungrammatical."""
    assert len(_parse(hf_sentence, hf_cfg(lexicalized=True))) == len(hi_sources)
    for hi_sentence in hi_sources:
        (tree,) = _parse(hi_sentence, hi_cfg(lexicalized=True))
        assert " ".join(hi_to_hf(tree).leaves()) == hf_sentence


@pytest.mark.parametrize(
    "sentence,expected",
    [
        ("cats that pursue dog swim", 1),
        ("cats that pursue dogs swim", 0),
        ("cats that dog hugs tease bird", 1),
        ("cat that dogs hug quickly sees birds", 1),
        ("cat chases dogs that swim", 0),
        ("cats chase dog that birds fear", 0),
        ("cats that swim sleep", 0),
    ],
)
def test_attractor_counts(sentence, expected):
    (tree,) = _parse(sentence, hi_cfg(lexicalized=True))
    assert n_attractors(tree) == expected
