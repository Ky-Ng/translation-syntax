"""Word-level tokenization and batching for the ce-rb training runs.

Sequence format (README.md "Dataset Format and Creation"):
    <bos> src <copy> <sep> src <eos>        (copy rows)
    <bos> src <translate> <sep> tgt <eos>   (translate rows)

The vocabulary is a grammar's lexicon plus the special tokens, word-level.
Each grammar has its own vocabulary (GRAMMAR_LEXICONS): Grammar V1
(src/cfg/grammar.py, 105 words) is the default, so every existing V1 run keeps
its token ids, and the number-agreement grammar
(src/cfg/grammar_number_agreement.py, 183 words) is selected with
`select_grammar("number_agreement")` before anything is encoded; `grammar_for`
derives the grammar from a dataset name. The special tokens come first in every
vocabulary, so <pad>, <bos>, <eos> and the task tokens share their ids across
grammars. Absolute-position shifting (README item 6): each sequence
is prepended with a random number of <pad> tokens in [0, MAX_SHIFT], so the
content occupies shifted absolute positions; pads are excluded from attention
and loss. All sequences are right-padded to CONTEXT_LEN.
"""

from pathlib import Path

import torch
from cfg import grammar, grammar_number_agreement

SPECIALS = ("<pad>", "<bos>", "<eos>", "<sep>", "<copy>", "<translate>")
GRAMMAR_LEXICONS = {"v1": grammar.LEXICON, "number_agreement": grammar_number_agreement.LEXICON}
DEFAULT_GRAMMAR = "v1"


def build_vocab(lexicon):
    """Special tokens first, then the lexicon's words sorted."""
    return list(SPECIALS) + sorted(word for words in lexicon.values() for word in words)


VOCAB = build_vocab(GRAMMAR_LEXICONS[DEFAULT_GRAMMAR])
STOI = {word: i for i, word in enumerate(VOCAB)}
PAD_ID = STOI["<pad>"]
_ACTIVE = {"grammar": DEFAULT_GRAMMAR}


def active_grammar():
    return _ACTIVE["grammar"]


def grammar_for(dataset):
    """Grammar of a dataset from its name: repos / directories with
    "number-agreement" in the name use the number-agreement grammar, everything
    else Grammar V1."""
    return "number_agreement" if "number-agreement" in Path(dataset).name else DEFAULT_GRAMMAR


def select_grammar(name):
    """Switch the active vocabulary to `name`'s grammar, in place, so that every
    `from training.data import VOCAB, STOI` binding (training.train,
    training.evaluate) sees the change. Call once, before encoding anything.
    Returns the vocabulary size."""
    vocab = build_vocab(GRAMMAR_LEXICONS[name])
    VOCAB[:] = vocab
    STOI.clear()
    STOI.update({word: i for i, word in enumerate(vocab)})
    _ACTIVE["grammar"] = name
    assert STOI["<pad>"] == PAD_ID
    return len(VOCAB)

MAX_SHIFT = 25
CONTEXT_LEN = 64  # max content is 20 tokens (8 src + 8 tgt + 4 specials) + 25 shift

DEFAULT_DATASET = "kylelovesllms/grammar-v1-frames-ce-rb"


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


def load_splits(dataset=DEFAULT_DATASET):
    """DatasetDict for the ce-rb data: a HF hub repo id, or a local directory
    holding train.csv / val.csv / test.csv (missing files are skipped)."""
    from datasets import load_dataset

    local = Path(dataset)
    if local.is_dir():
        names = {"train": "train.csv", "validation": "val.csv", "test": "test.csv"}
        files = {split: str(local / name) for split, name in names.items() if (local / name).exists()}
        return load_dataset("csv", data_files=files)
    return load_dataset(dataset)
