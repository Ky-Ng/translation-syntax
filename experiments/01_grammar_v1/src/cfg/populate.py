"""Populate the depth-1 syntactic frames with sampled lexical items (ce-rb dataset).

Second step of the dataset generation process (see DataGenFlow.png): each of
the 40 frames is populated with `--multiplier` distinct lexicalizations per
task, and (frame, task) pairs are partitioned into splits per README.md and
DataGenFlow.png:

    ambiguous frames    -> 90% train / 10% val, each with copy and translate
    unambiguous frames  -> train.copy and test.translate

The 90/10 carve is per (frame, task) group of examples, so every ambiguous
frame appears in both train and val. With the default multiplier of 100 this
yields 1,080 train + 120 val translate examples, 1,080 train + 120 val
ambiguous copy examples, 2,800 unambiguous train copy examples, and 2,800
test (translate) examples: 8,000 rows total.

Per frame, 2 * multiplier distinct lexicalizations are sampled with a seeded
RNG; the first half become copy examples and the second half translate
examples, so the two tasks never share a sentence within a frame (in
particular, unambiguous test translate sentences never appear as train copy
sentences). Copy rows use the HI frame/sentence as their own target; translate
rows carry the mirrored HF frame and the HF sentence obtained by applying the
frame's HI->HF token permutation to the sampled words.

Run with:
    cd experiments/01_grammar_v1/src && uv run python -m cfg.populate [--push USER/REPO]
"""

import argparse
import csv
import random
from pathlib import Path

from .frames import counts_table, single_rc_records
from .grammar import LEXICON
from .hub import PROJECT_ROOT, push_dataset

DEFAULT_MULTIPLIER = 100
DEFAULT_SEED = 0
DEFAULT_VAL_FRAC = 0.1
DEFAULT_REPO_NAME = "grammar-v1-frames-ce-rb"

CSV_COLUMNS = [
    "Split",
    "Task",
    "Attachment Site",
    "Gap Type",
    "HI Frame",
    "HF Frame",
    "HI Sentence",
    "HF Sentence",
    "is_ambiguous",
    "n_tokens",
    "depth",
]


def split_for(record, task, index, multiplier, n_val):
    """Partition per DataGenFlow.png: unambiguous translations are the test
    set, unambiguous copies all train, and ambiguous examples split 90/10
    into train/val (the last n_val of each (frame, task) group go to val)."""
    if not record["is_ambiguous"]:
        return "test" if task == "translate" else "train"
    return "val" if index >= multiplier - n_val else "train"


def sample_lexicalizations(categories, n, rng):
    """n distinct word tuples for a frame's category slots (rejection sampling;
    the smallest frame space is 26*26 = 676 >> 2 * multiplier)."""
    seen = set()
    out = []
    while len(out) < n:
        words = tuple(rng.choice(LEXICON[category]) for category in categories)
        if words not in seen:
            seen.add(words)
            out.append(words)
    return out


def build_examples(multiplier=DEFAULT_MULTIPLIER, seed=DEFAULT_SEED, val_frac=DEFAULT_VAL_FRAC):
    """All populated example rows, deterministic in (multiplier, seed, val_frac)."""
    rng = random.Random(seed)
    n_val = round(multiplier * val_frac)
    rows = []
    for record in single_rc_records():
        categories = record["hi_frame"].split()
        lexicalizations = sample_lexicalizations(categories, 2 * multiplier, rng)
        for task, chunk in (("copy", lexicalizations[:multiplier]), ("translate", lexicalizations[multiplier:])):
            for index, words in enumerate(chunk):
                hi_sentence = " ".join(words)
                if task == "copy":
                    hf_frame, hf_sentence = record["hi_bracketed"], hi_sentence
                else:
                    hf_frame = record["hf_bracketed"]
                    hf_sentence = " ".join(words[position - 1] for position in record["hf_order"])
                rows.append(
                    {
                        "Split": split_for(record, task, index, multiplier, n_val),
                        "Task": task,
                        "Attachment Site": record["attachment"],
                        "Gap Type": record["gap"],
                        "HI Frame": record["hi_bracketed"],
                        "HF Frame": hf_frame,
                        "HI Sentence": hi_sentence,
                        "HF Sentence": hf_sentence,
                        "is_ambiguous": record["is_ambiguous"],
                        "n_tokens": record["n_tokens"],
                        "depth": record["depth"],
                    }
                )
    return rows


def dataset_card(rows, multiplier, seed):
    train = [r for r in rows if r["Split"] == "train"]
    val = [r for r in rows if r["Split"] == "val"]
    test = [r for r in rows if r["Split"] == "test"]
    n_train_translate = sum(r["Task"] == "translate" for r in train)
    n_train_copy_ambiguous = sum(r["Task"] == "copy" and r["is_ambiguous"] for r in train)
    n_train_copy_unambiguous = sum(r["Task"] == "copy" and not r["is_ambiguous"] for r in train)
    n_val_translate = sum(r["Task"] == "translate" for r in val)
    return "\n".join(
        [
            "---",
            "pretty_name: Grammar V1 CE-RB (center embedding / right branching)",
            "configs:",
            "- config_name: default",
            "  data_files:",
            "  - split: train",
            "    path: data/train.csv",
            "  - split: validation",
            "    path: data/val.csv",
            "  - split: test",
            "    path: data/test.csv",
            "---",
            "",
            "# Grammar V1 — CE-RB Populated Frames",
            "",
            "The 40 depth-1 syntactic frames of "
            "[grammar-v1-frames](https://huggingface.co/datasets/kylelovesllms/grammar-v1-frames) "
            f"populated with {multiplier} distinct randomly sampled lexicalizations per (frame, task) "
            f"(seeded RNG, seed={seed}; grammar source: `experiments/01_grammar_v1/src/cfg/grammar.py`).",
            "",
            "Partition (README.md / DataGenFlow.png): ambiguous (in-distribution) frames contribute "
            "copy and translate examples split 90/10 into `train`/`validation` (per frame and task, "
            "so every ambiguous frame appears in both); unambiguous (out-of-distribution) frames "
            "contribute copy examples to `train` and translate examples to `test`. Copy and "
            "translate rows of a frame never share a sentence, so no test translate input appears "
            "as a train copy input.",
            "",
            f"- train: {len(train)} rows = {n_train_translate} translate (ambiguous) "
            f"+ {n_train_copy_ambiguous} copy (ambiguous) + {n_train_copy_unambiguous} copy (unambiguous)",
            f"- validation: {len(val)} rows = {n_val_translate} translate + {len(val) - n_val_translate} copy "
            "(ambiguous only)",
            f"- test: {len(test)} rows (translate, unambiguous)",
            "",
            "`HF Sentence` is the task target: the HI sentence itself for `copy` rows, the "
            "hierarchical (Head Final) translation for `translate` rows. Frames use indexed "
            "bracketed notation where `[N_3]` means the terminal filled by HI token 3.",
            "",
            "## Frame counts per category (before population)",
            "",
            *counts_table(single_rc_records()),
            "",
        ]
    )


def write_split_csvs(rows, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(r for r in rows if r["Split"] == split)
        paths[f"data/{split}.csv"] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--multiplier", type=int, default=DEFAULT_MULTIPLIER, help="examples per (frame, task)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC, help="ambiguous train/val carve")
    parser.add_argument(
        "--staging",
        type=Path,
        default=PROJECT_ROOT / "data" / DEFAULT_REPO_NAME,
        help="local staging dir for the split CSVs (per CLAUDE.md: ./data)",
    )
    parser.add_argument("--push", metavar="USER/REPO", help="push splits + dataset card to this HF dataset repo")
    args = parser.parse_args()

    rows = build_examples(args.multiplier, args.seed, args.val_frac)
    paths = write_split_csvs(rows, args.staging)
    for path in paths.values():
        print(f"wrote {path}")
    if args.push:
        card = dataset_card(rows, args.multiplier, args.seed)
        print(f"pushed to {push_dataset(args.push, paths, card)}")


if __name__ == "__main__":
    main()
