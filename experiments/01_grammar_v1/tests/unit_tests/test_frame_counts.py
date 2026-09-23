"""Closed-form verification of the depth-1 single-RC frame counts (src/cfg/frames.py).

How the closed form is derived — a product rule over independent choices:

1.  VP shapes with bare-noun arguments (no RC inside): a VP picks a verb arity
    and whether the verb carries an adverb (adverbs cannot recurse), so
        I = 2 intransitive shapes  (V_intrans | V_intrans Adv)
        T = 2 transitive shapes    (V_trans N | V_trans Adv N)
        V = I + T = 4 VP shapes
2.  RC shapes at depth 1 (embedded arguments are bare nouns):
        subject gap = C + any VP shape          -> R_s = V = 4
        object gap  = C + N + (V_trans +- Adv)  -> R_o = T = 2
3.  Attaching exactly one RC:
        no RC:    one frame per VP shape                      -> V
        subject:  every matrix VP shape has a subject slot    -> V * (R_s + R_o)
        object:   only transitive matrices have an object     -> T * (R_s + R_o)
4.  Ambiguity: the linear reversal rule agrees with the hierarchical
    translation iff the matrix subject is a bare noun AND every binary node
    reverses between HI and HF. Subject attachment violates the first
    condition; object gaps violate the second (S_obj_gap never reverses). So
        ambiguous   = V + T * R_s   (no-RC frames + object-attached subject gaps)
        unambiguous = everything else

The tests assert that the exhaustive enumeration in cfg.frames reproduces
these counts exactly, per category and per ambiguity class.
"""

from collections import Counter

from cfg.frames import CATEGORIES, single_rc_records, task_rows

I, T = 2, 2  # intransitive / transitive VP shapes (verb x optional adverb)
V = I + T  # all VP shapes
R_s, R_o = V, T  # subject-gap / object-gap RC shapes

EXPECTED_COUNTS = {
    ("None", "None"): V,  # 4
    ("Subject", "Subject Gap"): V * R_s,  # 16
    ("Subject", "Object Gap"): V * R_o,  # 8
    ("Object", "Subject Gap"): T * R_s,  # 8
    ("Object", "Object Gap"): T * R_o,  # 4
}
EXPECTED_AMBIGUOUS = V + T * R_s  # 12
FULLY_AMBIGUOUS_CATEGORIES = {("None", "None"), ("Object", "Subject Gap")}


def test_counts_per_category_match_closed_form():
    counts = Counter((r["attachment"], r["gap"]) for r in single_rc_records())
    assert counts == Counter(EXPECTED_COUNTS)
    assert sum(counts.values()) == V + (V + T) * (R_s + R_o)  # 40


def test_ambiguous_split_matches_closed_form():
    records = single_rc_records()
    assert sum(r["is_ambiguous"] for r in records) == EXPECTED_AMBIGUOUS
    assert sum(not r["is_ambiguous"] for r in records) == sum(EXPECTED_COUNTS.values()) - EXPECTED_AMBIGUOUS
    # ambiguity is decided entirely by the category, so each category is all-or-nothing
    for category in CATEGORIES:
        rows = [r for r in records if (r["attachment"], r["gap"]) == category]
        expected = category in FULLY_AMBIGUOUS_CATEGORIES
        assert all(r["is_ambiguous"] == expected for r in rows), category


def test_frames_are_unique():
    records = single_rc_records()
    assert len({r["hi_frame"] for r in records}) == len(records)
    assert len({r["hi_bracketed"] for r in records}) == len(records)


def test_task_rows_expand_each_frame_into_copy_and_translate():
    records = single_rc_records()
    rows = task_rows(records)
    assert len(rows) == 2 * len(records)
    for row in rows:
        assert row["Task"] in ("copy", "translate")
    copy_rows = [row for row in rows if row["Task"] == "copy"]
    assert all(row["HF Frame"] == row["HI Frame"] for row in copy_rows), "copy target is the HI frame itself"
