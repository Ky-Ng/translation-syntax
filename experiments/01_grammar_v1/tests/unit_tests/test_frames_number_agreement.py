"""Closed-form verification of the number-agreement frame inventory
(src/cfg/frames_number_agreement.py).

Every V1 single-RC frame (closed form in test_frame_counts.py) has 2^k number
variants, k its number of nouns: each noun is sg or pl freely, the verbs are
then fixed by agreement. Counting the V1 frames of each category by k:

    None / None            k=1: 2 (V_intrans +- Adv)   k=2: 2 (V_trans +- Adv)   -> 2*2 + 2*4        = 12
    Subject / Subject Gap  k=1: 4   k=2: 8   k=3: 4                             -> 4*2 + 8*4 + 4*8  = 72
    Subject / Object Gap   k=2: 4   k=3: 4                                      -> 4*4 + 4*8        = 48
    Object / Subject Gap   k=2: 4   k=3: 4                                      -> 4*4 + 4*8        = 48
    Object / Object Gap    k=3: 4                                               -> 4*8              = 32

Ambiguity is decided by the V1 frame (number features only reorder like any
token), so ambiguous = None/None + Object/Subject Gap = 12 + 48 = 60.

Attractors: a noun of the other number between a verb and its subject exists
iff the RC is subject-attached and contains a noun (the RC object of a subject
gap, the embedded subject of an object gap) whose number differs from the
matrix subject's. That is half of the Subject/Subject Gap frames with a
transitive RC (8 V1 frames: 4 with k=2, 4 with k=3 -> (4*4 + 4*8) / 2 = 24)
and half of the Subject/Object Gap frames (48 / 2 = 24): 48 frames, each with
exactly one attractor at depth 1.
"""

import re
from collections import Counter

from cfg import frames as v1_frames
from cfg.frames_number_agreement import single_rc_records, task_rows
from cfg.grammar_number_agreement import LEXICON, NUMBERS, VERB_CATEGORIES, number_of, strip_number

EXPECTED_COUNTS = {
    ("None", "None"): 12,
    ("Subject", "Subject Gap"): 72,
    ("Subject", "Object Gap"): 48,
    ("Object", "Subject Gap"): 48,
    ("Object", "Object Gap"): 32,
}
EXPECTED_AMBIGUOUS = 60
EXPECTED_ATTRACTOR_FRAMES = {("Subject", "Subject Gap"): 24, ("Subject", "Object Gap"): 24}


def test_counts_per_category_match_closed_form():
    counts = Counter((r["attachment"], r["gap"]) for r in single_rc_records())
    assert counts == Counter(EXPECTED_COUNTS)
    assert sum(counts.values()) == 212


def test_number_variants_of_each_v1_frame():
    """The records project onto exactly the 40 V1 frames, each 2^k times."""
    records = single_rc_records()
    projected = Counter(r["v1_frame"] for r in records)
    v1_records = {r["hi_frame"]: r for r in v1_frames.single_rc_records()}
    assert set(projected) == set(v1_records)
    for frame, n in projected.items():
        assert n == 2 ** frame.split().count("N"), frame
    # the category and ambiguity of a variant are those of its V1 frame
    for r in records:
        v1_record = v1_records[r["v1_frame"]]
        assert (r["attachment"], r["gap"], r["is_ambiguous"]) == (v1_record["attachment"], v1_record["gap"], v1_record["is_ambiguous"])
        assert " ".join(map(strip_number, r["hf_frame"].split())) == v1_record["hf_frame"]
        assert r["hf_order"] == v1_record["hf_order"]


def test_ambiguous_split_matches_closed_form():
    records = single_rc_records()
    assert sum(r["is_ambiguous"] for r in records) == EXPECTED_AMBIGUOUS


def test_agreement_columns():
    """Matrix Number is the first token's number; RC Number equals the matrix
    number for every subject-attached subject gap (same controller) and
    for exactly half of the other RC frames; only center-embedded frames
    have attractors, one each, in the closed-form counts."""
    records = single_rc_records()
    attractor_frames = Counter()
    for r in records:
        assert r["matrix_number"] == number_of(r["hi_frame"].split()[0])
        assert r["n_attractors"] in (0, 1)
        if r["n_attractors"]:
            attractor_frames[(r["attachment"], r["gap"])] += 1
        if r["gap"] == "None":
            assert r["rc_number"] == "None" and r["n_attractors"] == 0
        elif (r["attachment"], r["gap"]) == ("Subject", "Subject Gap"):
            assert r["rc_number"] == r["matrix_number"]
    assert attractor_frames == Counter(EXPECTED_ATTRACTOR_FRAMES)
    for category in (("Subject", "Object Gap"), ("Object", "Subject Gap"), ("Object", "Object Gap")):
        rows = [r for r in records if (r["attachment"], r["gap"]) == category]
        assert sum(r["rc_number"] == r["matrix_number"] for r in rows) == len(rows) // 2, category
    # an attractor is a noun of the other number between the matrix subject
    # (token 0) and the matrix verb, which in a subject-attached frame is the
    # last verb; object-attached frames have nothing between the two
    for r in records:
        tokens = r["hi_frame"].split()
        matrix_verb = max(i for i, t in enumerate(tokens) if strip_number(t) in VERB_CATEGORIES)
        intervening = [t for t in tokens[1:matrix_verb] if strip_number(t) == "N" and number_of(t) != r["matrix_number"]]
        assert bool(r["n_attractors"]) == (r["attachment"] == "Subject" and bool(intervening)), r["hi_frame"]


def test_examples_are_lexicalizations_of_their_frames():
    """The stock examples fill each slot with a word of the slot's category and
    number; the HF example is the HI example under the frame's permutation."""
    records = single_rc_records()
    for r in records:
        tokens, words = r["hi_frame"].split(), r["hi_example"].split()
        assert len(tokens) == len(words) == r["n_tokens"]
        for token, word in zip(tokens, words):
            assert word in LEXICON[token], (token, word)
        assert r["hf_example"].split() == [words[position - 1] for position in r["hf_order"]]
    # the stock lemmas are cat/dog/bird, pursue/hug, swim/sleep in order of appearance
    examples = {r["hi_example"]: r["hf_example"] for r in records}
    assert examples["cats that pursue dog swim"] == "dog pursue that cats swim"
    assert examples["cats that dog pursues hug bird"] == "dog pursues that cats bird hug"
    assert examples["cat chases dogs that swim"] == "cat swim that dogs chases" if "cat chases dogs that swim" in examples else True
    assert examples["cat pursues dogs that swim"] == "cat swim that dogs pursues"


def test_frames_are_unique_and_sorted_singular_first():
    records = single_rc_records()
    assert len({r["hi_frame"] for r in records}) == len(records)
    assert len({r["hi_bracketed"] for r in records}) == len(records)
    assert records[0]["hi_frame"] == "N_sg V_intrans_sg" and records[1]["hi_frame"] == "N_pl V_intrans_pl"
    assert all(re.fullmatch(r"(N|V_intrans|V_trans)_(sg|pl)|Adv|C", t) for r in records for t in r["hi_frame"].split())
    assert set(NUMBERS) == {r["matrix_number"] for r in records}


def test_task_rows_expand_each_frame_into_copy_and_translate():
    records = single_rc_records()
    rows = task_rows(records)
    assert len(rows) == 2 * len(records) == 424
    copy_rows = [row for row in rows if row["Task"] == "copy"]
    assert all(row["HF Frame"] == row["HI Frame"] for row in copy_rows)
    for row in rows:
        assert re.findall(r"\[((?:N|V_intrans|V_trans)_(?:sg|pl)|Adv|C)_\d+\]", row["HI Frame"])
