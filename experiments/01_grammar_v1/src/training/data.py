"""Word-level tokenization and batching for the ce-rb training runs.

Sequence format (README.md "Dataset Format and Creation"):
    <bos> src <copy> <sep> src <eos>        (copy rows)
    <bos> src <translate> <sep> tgt <eos>   (translate rows)

The vocabulary is the grammar lexicon (src/cfg/grammar.py) plus the special
tokens, word-level. Absolute-position shifting (README item 6): each sequence
is prepended with a random number of <pad> tokens in [0, MAX_SHIFT], so the
content occupies shifted absolute positions; pads are excluded from attention
and loss. All sequences are right-padded to CONTEXT_LEN.
"""

import torch

from cfg.grammar import LEXICON

SPECIALS = ("<pad>", "<bos>", "<eos>", "<sep>", "<copy>", "<translate>")
VOCAB = list(SPECIALS) + sorted(word for words in LEXICON.values() for word in words)
STOI = {word: i for i, word in enumerate(VOCAB)}
PAD_ID = STOI["<pad>"]

MAX_SHIFT = 25
CONTEXT_LEN = 64  # max content is 20 tokens (8 src + 8 tgt + 4 specials) + 25 shift


def encode_row(task, source, target):
    """Token ids for one dataset row; `task` is 'copy' or 'translate'."""
    words = ["<bos>", *source.split(), f"<{task}>", "<sep>", *target.split(), "<eos>"]
    return [STOI[word] for word in words]


def encode_split(split):
    """Encode a HuggingFace split of the ce-rb dataset (columns from cfg.populate)."""
    return [encode_row(row["Task"], row["HI Sentence"], row["HF Sentence"]) for row in split]


def collate(sequences, rng):
    """Batch encoded sequences with random position shift via left-padding.

    Returns input_ids, attention_mask (both [B, CONTEXT_LEN]) and labels
    ([B, CONTEXT_LEN], -100 on every pad position).
    """
    batch = len(sequences)
    input_ids = torch.full((batch, CONTEXT_LEN), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((batch, CONTEXT_LEN), dtype=torch.long)
    labels = torch.full((batch, CONTEXT_LEN), -100, dtype=torch.long)
    for i, seq in enumerate(sequences):
        shift = rng.randint(0, MAX_SHIFT)
        ids = torch.tensor(seq, dtype=torch.long)
        input_ids[i, shift : shift + len(seq)] = ids
        attention_mask[i, shift : shift + len(seq)] = 1
        labels[i, shift : shift + len(seq)] = ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
