"""Tests for training.data — run these on the cluster before launching the sweep."""

import random

from cfg.grammar import LEXICON
from training.data import CONTEXT_LEN, MAX_SHIFT, PAD_ID, STOI, VOCAB, collate, encode_row


def test_vocab():
    assert len(VOCAB) == 6 + sum(len(words) for words in LEXICON.values()) == 111
    assert len(set(VOCAB)) == len(VOCAB)
    assert VOCAB[PAD_ID] == "<pad>" and PAD_ID == 0


def test_encode_row_roundtrip():
    ids = encode_row("translate", "cat chases dog", "cat dog chases")
    assert [VOCAB[i] for i in ids] == ["<bos>", "cat", "chases", "dog", "<translate>", "<sep>", "cat", "dog", "chases", "<eos>"]
    ids = encode_row("copy", "dog swims", "dog swims")
    assert [VOCAB[i] for i in ids] == ["<bos>", "dog", "swims", "<copy>", "<sep>", "dog", "swims", "<eos>"]


def test_longest_sequence_fits_with_max_shift():
    # longest ce-rb sentence is 8 tokens -> 8 + 8 + 4 specials = 20 content tokens
    longest = encode_row("translate", "a b c d e f g h".replace("a", "cat"), "x " * 7 + "x")
    assert len(longest) == 20
    assert 20 + MAX_SHIFT <= CONTEXT_LEN


def test_collate_shift_masking_and_content():
    seqs = [
        encode_row("copy", "dog swims", "dog swims"),
        encode_row("translate", "cat chases dog", "cat dog chases"),
    ]
    batch = collate(seqs, random.Random(0))
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape == (2, CONTEXT_LEN)
    for i, seq in enumerate(seqs):
        attn = batch["attention_mask"][i]
        (positions,) = attn.nonzero(as_tuple=True)
        shift = positions[0].item()
        assert 0 <= shift <= MAX_SHIFT
        assert positions.tolist() == list(range(shift, shift + len(seq))), "content must be contiguous"
        assert batch["input_ids"][i, shift : shift + len(seq)].tolist() == seq
        # loss and attention are masked exactly on the pads
        assert (batch["labels"][i] == -100).tolist() == (attn == 0).tolist()
        assert (batch["input_ids"][i][attn == 0] == PAD_ID).all()


def test_collate_shifts_vary():
    seqs = [encode_row("copy", "dog swims", "dog swims")] * 50
    batch = collate(seqs, random.Random(0))
    shifts = {row.nonzero(as_tuple=True)[0][0].item() for row in batch["attention_mask"]}
    assert len(shifts) > 5, "position shifts should vary across a batch"
