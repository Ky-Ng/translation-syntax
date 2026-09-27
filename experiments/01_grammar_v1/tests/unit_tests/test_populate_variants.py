"""Tests for the copy-task dataset variants ce / rb of src/cfg/populate.py.

Closed form of the ce-rb train token budget at multiplier 100. A train row
encodes to 2 * n_tokens + 4 non-pad tokens (<bos> src <task> <sep> tgt <eos>),
so a (task, category) group with m rows per frame contributes
m * (2 * S + 4 * F) tokens, where F is the number of frames in the category
and S the sum of their n_tokens (from the frame inventory, test_frame_counts.py
derives F; S follows from the VP shapes: verb, verb+Adv, verb+N, verb+Adv+N):

    category            F    S
    None/None           4    12
    Subject/SubjectGap 16    96
    Subject/ObjectGap   8    52
    Object/SubjectGap   8    52
    Object/ObjectGap    4    28

ce-rb train = translate(ambiguous, 90/frame) + copy(ambiguous, 90/frame)
            + copy(unambiguous, 100/frame) = 78,080 tokens.
A variant keeps the translate rows and only the copy rows of its attachment
sites, then boosts every remaining train group by

    boost = 78,080 / (tokens left after dropping the copy rows)

so ce (no-RC + Subject copies) = 78,080 / 58,640 and rb (no-RC + Object copies)
= 78,080 / 38,880. Val and test are byte-identical to ce-rb.
"""

import re

import pytest

from cfg.grammar import LEXICON
from cfg.populate import (
    ATTACHMENT_SITES,
    COPY_VARIANTS,
    build_examples,
    sequence_tokens,
    solve_boost,
    train_breakdown,
    train_tokens,
)

MULTIPLIER = 100
N_TRAIN_AMBIGUOUS, N_TRAIN_UNAMBIGUOUS = 90, 100

# category -> (frames, sum of n_tokens)
INVENTORY = {
    ("None", "None"): (4, 12),
    ("Subject", "Subject Gap"): (16, 96),
    ("Subject", "Object Gap"): (8, 52),
    ("Object", "Subject Gap"): (8, 52),
    ("Object", "Object Gap"): (4, 28),
}
AMBIGUOUS = {("None", "None"), ("Object", "Subject Gap")}


def group_tokens(category, rows_per_frame):
    frames, n_tokens = INVENTORY[category]
    return rows_per_frame * (2 * n_tokens + 4 * frames)


def closed_form_tokens(copy_attachments):
    """Train tokens of a variant before boosting."""
    total = 0
    for category in INVENTORY:
        ambiguous = category in AMBIGUOUS
        if ambiguous:
            total += group_tokens(category, N_TRAIN_AMBIGUOUS)  # translate
        if category[0] in copy_attachments:
            total += group_tokens(category, N_TRAIN_AMBIGUOUS if ambiguous else N_TRAIN_UNAMBIGUOUS)
    return total


BASELINE_TOKENS = closed_form_tokens(ATTACHMENT_SITES)


def key(row):
    return (row["Split"], row["Task"], row["HI Frame"], row["HI Sentence"], row["HF Sentence"])


@pytest.fixture(scope="module")
def baseline():
    return build_examples(MULTIPLIER, 0)


@pytest.fixture(scope="module", params=["ce", "rb"])
def variant(request):
    attachments = COPY_VARIANTS[request.param]
    boost = solve_boost(attachments, MULTIPLIER, 0)
    return request.param, attachments, boost, build_examples(MULTIPLIER, 0, copy_attachments=attachments, boost=boost)


def test_closed_form_budget_and_factors():
    assert BASELINE_TOKENS == 78_080
    assert closed_form_tokens(COPY_VARIANTS["ce"]) == 58_640
    assert closed_form_tokens(COPY_VARIANTS["rb"]) == 38_880
    assert solve_boost(COPY_VARIANTS["ce"], MULTIPLIER, 0) == pytest.approx(78_080 / 58_640)
    assert solve_boost(COPY_VARIANTS["rb"], MULTIPLIER, 0) == pytest.approx(78_080 / 38_880)
    # stricter reading that also drops the no-RC copy rows
    assert solve_boost(("Subject",), MULTIPLIER, 0) == pytest.approx(78_080 / 55_040)
    assert solve_boost(("Object",), MULTIPLIER, 0) == pytest.approx(78_080 / 35_280)


def test_baseline_variant_is_the_cerb_dataset(baseline):
    assert train_tokens(baseline) == BASELINE_TOKENS
    assert solve_boost(COPY_VARIANTS["ce-rb"], MULTIPLIER, 0) == 1.0
    assert build_examples(MULTIPLIER, 0, copy_attachments=COPY_VARIANTS["ce-rb"], boost=1.0) == baseline


def test_boost_below_one_is_rejected():
    with pytest.raises(ValueError):
        build_examples(3, 0, boost=0.5)


def test_val_and_test_identical_to_baseline(baseline, variant):
    _, _, _, rows = variant
    for split in ("val", "test"):
        assert [r for r in rows if r["Split"] == split] == [r for r in baseline if r["Split"] == split], split


def test_train_keeps_only_allowed_copy_structures(variant):
    _, attachments, _, rows = variant
    train = [r for r in rows if r["Split"] == "train"]
    for r in train:
        if r["Task"] == "copy":
            assert r["Attachment Site"] in attachments
        else:
            assert r["is_ambiguous"], "train translate rows are the ambiguous frames only"


def test_train_is_baseline_train_minus_dropped_copies_plus_extras(baseline, variant):
    _, attachments, _, rows = variant
    base_train = {key(r) for r in baseline if r["Split"] == "train"}
    kept_base = {k for k in base_train if not (k[1] == "copy" and _attachment_of(baseline, k) not in attachments)}
    train = {key(r) for r in rows if r["Split"] == "train"}
    assert kept_base <= train, "every kept baseline train row survives"
    assert not (train & (base_train - kept_base)), "dropped copy rows do not come back"
    assert len(train) == sum(r["Split"] == "train" for r in rows), "no duplicate train rows"


def _attachment_of(rows, k):
    for r in rows:
        if key(r) == k:
            return r["Attachment Site"]
    raise KeyError(k)


def test_every_remaining_group_boosted_by_the_same_factor(baseline, variant):
    _, attachments, boost, rows = variant
    assert boost > 1
    base_sizes = {}
    for r in baseline:
        if r["Split"] == "train" and (r["Task"] == "translate" or r["Attachment Site"] in attachments):
            base_sizes[(r["HI Frame"], r["Task"])] = base_sizes.get((r["HI Frame"], r["Task"]), 0) + 1
    sizes = {}
    for r in rows:
        if r["Split"] == "train":
            sizes[(r["HI Frame"], r["Task"])] = sizes.get((r["HI Frame"], r["Task"]), 0) + 1
    assert set(sizes) == set(base_sizes), "exactly the kept (frame, task) groups appear in train"
    for group, size in base_sizes.items():
        assert sizes[group] == round(size * boost), group


def test_token_budget_matches_baseline_up_to_per_group_rounding(baseline, variant):
    _, _, _, rows = variant
    n_groups = len({(r["HI Frame"], r["Task"]) for r in rows if r["Split"] == "train"})
    longest = max(sequence_tokens(r) for r in rows)
    tolerance = 0.5 * longest * n_groups  # at most half a row rounded per group
    assert abs(train_tokens(rows) - train_tokens(baseline)) <= tolerance
    assert abs(train_tokens(rows) - train_tokens(baseline)) / train_tokens(baseline) < 0.005


def test_lexicalizations_distinct_and_task_disjoint_within_frame(variant):
    _, _, _, rows = variant
    by_frame_task = {}
    for r in rows:
        by_frame_task.setdefault((r["HI Frame"], r["Task"]), []).append(r["HI Sentence"])
    for group, sentences in by_frame_task.items():
        assert len(set(sentences)) == len(sentences), f"duplicate lexicalization in {group}"
    for frame in {frame for frame, _ in by_frame_task}:
        copies = set(by_frame_task.get((frame, "copy"), []))
        translations = set(by_frame_task.get((frame, "translate"), []))
        assert not copies & translations, f"copy/translate share a sentence in {frame}"
    # in particular no train copy input is a test translate input
    test_inputs = {r["HI Sentence"] for r in rows if r["Split"] == "test"}
    train_copy_inputs = {r["HI Sentence"] for r in rows if r["Split"] == "train" and r["Task"] == "copy"}
    assert not test_inputs & train_copy_inputs


def test_extra_rows_are_well_formed(variant):
    _, _, _, rows = variant
    for r in rows:
        categories = re.findall(r"\[([A-Za-z_]+)_\d+\]", r["HI Frame"])
        words = r["HI Sentence"].split()
        assert len(words) == len(categories) == r["n_tokens"]
        for category, word in zip(categories, words):
            assert word in LEXICON[category], (category, word)
        target = r["HF Sentence"].split()
        if r["Task"] == "copy":
            assert r["HF Sentence"] == r["HI Sentence"] and r["HF Frame"] == r["HI Frame"]
        else:
            assert sorted(target) == sorted(words)
            assert ((words[:1] + words[:0:-1]) == target) == r["is_ambiguous"]


def test_train_breakdown_covers_train_rows(variant):
    _, _, _, rows = variant
    breakdown = train_breakdown(rows)
    assert sum(e["rows"] for e in breakdown.values()) == sum(r["Split"] == "train" for r in rows)
    assert sum(e["tokens"] for e in breakdown.values()) == train_tokens(rows)


def test_variants_deterministic_and_seed_sensitive():
    a = build_examples(4, 7, copy_attachments=("Subject",), boost=1.5)
    assert a == build_examples(4, 7, copy_attachments=("Subject",), boost=1.5)
    assert a != build_examples(4, 8, copy_attachments=("Subject",), boost=1.5)
