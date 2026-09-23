"""Tests for the ce-rb populated dataset (src/cfg/populate.py).

The required counts at multiplier 100 follow from the frame inventory
(12 ambiguous + 28 unambiguous frames) and the DataGenFlow partition,
with the ambiguous examples carved 90/10 into train/val per (frame, task):
    train.translate            = 12 * 90  = 1,080   (ambiguous only)
    val.translate              = 12 * 10  =   120
    train.copy   (ambiguous)   = 12 * 90  = 1,080
    val.copy                   = 12 * 10  =   120
    train.copy   (unambiguous) = 28 * 100 = 2,800
    test.translate             = 28 * 100 = 2,800   (unambiguous only)
"""

import pytest

from cfg.grammar import LEXICON
from cfg.populate import build_examples

MULTIPLIER = 100


@pytest.fixture(scope="module")
def rows():
    return build_examples(multiplier=MULTIPLIER, seed=0)


def test_required_split_counts(rows):
    train = [r for r in rows if r["Split"] == "train"]
    val = [r for r in rows if r["Split"] == "val"]
    test = [r for r in rows if r["Split"] == "test"]
    assert sum(r["Task"] == "translate" for r in train) == 1_080
    assert sum(r["Task"] == "copy" and r["is_ambiguous"] for r in train) == 1_080
    assert sum(r["Task"] == "copy" and not r["is_ambiguous"] for r in train) == 2_800
    assert sum(r["Task"] == "translate" for r in val) == 120
    assert sum(r["Task"] == "copy" for r in val) == 120
    assert len(test) == 2_800
    assert len(rows) == 8_000


def test_partition_matches_datagenflow(rows):
    for r in rows:
        if r["Split"] == "test":
            # held out: translations of unambiguous (OOD) frames only
            assert r["Task"] == "translate" and not r["is_ambiguous"]
        elif r["Split"] == "val":
            # val is the 10% carve of the ambiguous examples (both tasks)
            assert r["is_ambiguous"]
        else:
            # train: ambiguous copy+translate, unambiguous copy
            assert r["is_ambiguous"] or r["Task"] == "copy"


def test_val_carve_is_per_frame_and_task(rows):
    ambiguous = [r for r in rows if r["is_ambiguous"]]
    by_frame_task = {}
    for r in ambiguous:
        by_frame_task.setdefault((r["HI Frame"], r["Task"]), []).append(r["Split"])
    assert len(by_frame_task) == 24  # 12 ambiguous frames x 2 tasks
    for key, splits in by_frame_task.items():
        assert splits.count("train") == 90 and splits.count("val") == 10, key


def test_examples_per_frame_distinct_and_task_disjoint(rows):
    by_frame_task = {}
    for r in rows:
        by_frame_task.setdefault((r["HI Frame"], r["Task"]), []).append(r["HI Sentence"])
    for (frame, task), sentences in by_frame_task.items():
        assert len(sentences) == MULTIPLIER, (frame, task)
        assert len(set(sentences)) == MULTIPLIER, f"duplicate lexicalization in {frame} {task}"
    frames = {frame for frame, _ in by_frame_task}
    assert len(frames) == 40 and len(by_frame_task) == 80
    for frame in frames:
        copy_sents = set(by_frame_task[(frame, "copy")])
        translate_sents = set(by_frame_task[(frame, "translate")])
        assert not copy_sents & translate_sents, f"copy/translate share a sentence in {frame}"


def test_words_match_frame_slots(rows):
    import re

    for r in rows:
        categories = re.findall(r"\[([A-Za-z_]+)_\d+\]", r["HI Frame"])
        words = r["HI Sentence"].split()
        assert len(words) == len(categories) == r["n_tokens"]
        for category, word in zip(categories, words):
            assert word in LEXICON[category], (category, word)


def test_targets(rows):
    for r in rows:
        words = r["HI Sentence"].split()
        target = r["HF Sentence"].split()
        if r["Task"] == "copy":
            assert r["HF Sentence"] == r["HI Sentence"]
            assert r["HF Frame"] == r["HI Frame"]
        else:
            assert sorted(target) == sorted(words), "translation must only reorder words"
            linear = words[:1] + words[:0:-1]
            # ambiguity holds at the word level: linear reversal matches the
            # hierarchical target exactly for in-distribution rows
            assert (linear == target) == r["is_ambiguous"]


def test_deterministic_given_seed():
    a = build_examples(multiplier=3, seed=123)
    b = build_examples(multiplier=3, seed=123)
    assert a == b
    assert a != build_examples(multiplier=3, seed=124)
