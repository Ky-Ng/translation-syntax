"""Populate the depth-1 syntactic frames with sampled lexical items (ce-rb / ce / rb datasets).

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
test (translate) examples: 8,000 rows total (the ce-rb dataset).

Per frame, 2 * multiplier distinct lexicalizations are sampled with a seeded
RNG; the first half become copy examples and the second half translate
examples, so the two tasks never share a sentence within a frame (in
particular, unambiguous test translate sentences never appear as train copy
sentences). Copy rows use the HI frame/sentence as their own target; translate
rows carry the mirrored HF frame and the HF sentence obtained by applying the
frame's HI->HF token permutation to the sampled words.

Copy-task variants (README "Training Copying Syntactic Structures"): the
`--variant` flag restricts which frames contribute *train copy* rows, by the
attachment site of their relative clause:

    ce-rb   every frame (the baseline above)
    ce      simple (no RC) + center embedded: subject-attached RCs (Subject/Subject Gap, Subject/Object Gap)
    rb      simple (no RC) + right branching: object-attached RCs (Object/Subject Gap, Object/Object Gap)

Translate rows and the val/test splits are byte-identical across variants: the
base 2 * multiplier lexicalizations per frame are drawn from the same RNG
stream before any extra ones. To hold the train token budget fixed (the
non-pad tokens the model is trained on, `<bos> src <task> <sep> tgt <eos>` =
2 * n_tokens + 4 per row), every remaining train (frame, task) group is
boosted by the same factor -- the ce-rb budget divided by the budget left after
dropping the copy rows -- with additional distinct lexicalizations (rounded per
group). The factor is solved automatically (`--boost` overrides it) and
reported on stdout and in the dataset card.

Grammars (`--grammar`): `v1` (default) populates the 40 Grammar V1 frames as
above. `number_agreement` populates the 212 frames of
cfg.grammar_number_agreement (inventory: cfg.frames_number_agreement); its rows
carry the agreement columns Matrix Number, RC Number and n_attractors, and its
datasets are named grammar-number-agreement-frames-<variant>. Because the
frame count differs, the multiplier of a non-v1 grammar is solved
(`solve_multiplier`) so that its ce-rb train token budget is closest to the
Grammar V1 ce-rb budget (78,080 tokens at multiplier 100; `--train-tokens`
overrides the target, `--multiplier` the result): for number_agreement this is
multiplier 18, 77,536 tokens (0.7% under; the per-group rounding of the boost
factor cannot close a gap smaller than one row per (frame, task) group). The
ce / rb variants of a grammar are then boosted to their own ce-rb budget, as
for v1; with every group at 16 or 18 rows the rounding no longer averages out
across groups, so their residuals are larger than v1's (ce +136, rb -592
tokens: 0.2% / 0.8%).

Run with:
    cd experiments/01_grammar_v1/src && uv run python -m cfg.populate [--grammar number_agreement] [--variant ce|rb] [--push USER/REPO]
"""

import argparse
import csv
import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from . import frames, frames_number_agreement, grammar, grammar_number_agreement
from .frames import CATEGORIES, counts_table
from .hub import PROJECT_ROOT, push_dataset

DEFAULT_MULTIPLIER = 100
DEFAULT_SEED = 0
DEFAULT_VAL_FRAC = 0.1
DEFAULT_VARIANT = "ce-rb"

ATTACHMENT_SITES = ("None", "Subject", "Object")

# variant -> attachment sites whose frames contribute train copy rows
COPY_VARIANTS = {
    "ce-rb": ATTACHMENT_SITES,
    "ce": ("None", "Subject"),
    "rb": ("None", "Object"),
}
VARIANT_DESCRIPTIONS = {
    "ce-rb": "copy every structure: center embedded and right branching",
    "ce": "copy simple (no RC) and center embedded (subject-attached RC) structures only",
    "rb": "copy simple (no RC) and right branching (object-attached RC) structures only",
}


@dataclass(frozen=True)
class GrammarSpec:
    """What populating needs to know about a grammar."""

    name: str
    records: Callable[[], list]  # depth-1 frame inventory (cfg.frames*.single_rc_records)
    lexicon: dict  # terminal category -> words
    source: str  # grammar module, for the dataset card
    frames_repo: str  # HF dataset holding the frame inventory
    repo_prefix: str  # populated datasets are named f"{repo_prefix}-{variant}"
    title: str  # dataset card title
    extra_columns: tuple = ()  # (record key, CSV column) pairs carried into every row
    notes: str = ""  # dataset card paragraph describing the grammar's extra columns


GRAMMARS = {
    "v1": GrammarSpec(
        name="v1",
        records=frames.single_rc_records,
        lexicon=grammar.LEXICON,
        source="experiments/01_grammar_v1/src/cfg/grammar.py",
        frames_repo="kylelovesllms/grammar-v1-frames",
        repo_prefix="grammar-v1-frames",
        title="Grammar V1",
    ),
    "number_agreement": GrammarSpec(
        name="number_agreement",
        records=frames_number_agreement.single_rc_records,
        lexicon=grammar_number_agreement.LEXICON,
        source="experiments/01_grammar_v1/src/cfg/grammar_number_agreement.py",
        frames_repo="kylelovesllms/grammar-number-agreement-frames",
        repo_prefix="grammar-number-agreement-frames",
        title="Grammar V1 + Number Agreement",
        extra_columns=(("matrix_number", "Matrix Number"), ("rc_number", "RC Number"), ("n_attractors", "n_attractors")),
        notes=(
            "Grammar V1 with subject-verb number agreement: nouns and verbs are singular or plural "
            "(`N_sg` / `N_pl`, `V_trans_sg` / `V_trans_pl`, ...), the matrix verb agrees with the matrix subject, "
            "a subject-gap RC verb with the head noun and an object-gap RC verb with the embedded subject, while "
            "objects take either number. `Matrix Number` is the number of the matrix subject, `RC Number` that of the "
            "noun the RC verb agrees with (`None` without an RC), and `n_attractors` counts nouns of the other number "
            "between a verb and the noun it agrees with (only center-embedded sentences have any). Every sentence is "
            "grammatical under agreement by construction: each frame slot fixes the number of the word drawn for it."
        ),
    ),
}
V1 = GRAMMARS["v1"]

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


def csv_columns(grammar_spec=V1):
    """CSV_COLUMNS with the grammar's extra columns inserted after Gap Type."""
    columns = list(CSV_COLUMNS)
    at = columns.index("Gap Type") + 1
    return columns[:at] + [column for _, column in grammar_spec.extra_columns] + columns[at:]


def repo_name(variant, grammar_spec=V1):
    return f"{grammar_spec.repo_prefix}-{variant}"


def split_for(record, task, index, multiplier, n_val):
    """Partition per DataGenFlow.png: unambiguous translations are the test
    set, unambiguous copies all train, and ambiguous examples split 90/10
    into train/val (the last n_val of each (frame, task) group go to val)."""
    if not record["is_ambiguous"]:
        return "test" if task == "translate" else "train"
    return "val" if index >= multiplier - n_val else "train"


def sample_lexicalizations(categories, n, rng, exclude=(), lexicon=None):
    """n distinct word tuples for a frame's category slots, none of them in
    `exclude` (rejection sampling; the smallest frame space is 26*26 = 676 >>
    2 * multiplier)."""
    lexicon = V1.lexicon if lexicon is None else lexicon
    seen = set(exclude)
    out = []
    while len(out) < n:
        words = tuple(rng.choice(lexicon[category]) for category in categories)
        if words not in seen:
            seen.add(words)
            out.append(words)
    return out


def train_group_size(record, task, multiplier, n_val, copy_attachments):
    """Base number of train rows of a (frame, task) group under a copy variant."""
    if task == "copy" and record["attachment"] not in copy_attachments:
        return 0
    if not record["is_ambiguous"]:
        return multiplier if task == "copy" else 0  # unambiguous translations are the test set
    return multiplier - n_val


def make_row(record, task, split, words, grammar_spec=V1):
    hi_sentence = " ".join(words)
    if task == "copy":
        hf_frame, hf_sentence = record["hi_bracketed"], hi_sentence
    else:
        hf_frame = record["hf_bracketed"]
        hf_sentence = " ".join(words[position - 1] for position in record["hf_order"])
    return {
        "Split": split,
        "Task": task,
        "Attachment Site": record["attachment"],
        "Gap Type": record["gap"],
        **{column: record[key] for key, column in grammar_spec.extra_columns},
        "HI Frame": record["hi_bracketed"],
        "HF Frame": hf_frame,
        "HI Sentence": hi_sentence,
        "HF Sentence": hf_sentence,
        "is_ambiguous": record["is_ambiguous"],
        "n_tokens": record["n_tokens"],
        "depth": record["depth"],
    }


def build_examples(
    multiplier=DEFAULT_MULTIPLIER,
    seed=DEFAULT_SEED,
    val_frac=DEFAULT_VAL_FRAC,
    copy_attachments=ATTACHMENT_SITES,
    boost=1.0,
    grammar_spec=V1,
):
    """All populated example rows, deterministic in the arguments.

    copy_attachments: attachment sites whose frames keep their train copy rows
    (val/test rows are never touched, so the validation split still holds copy
    rows of the dropped structures). boost: every remaining train (frame, task)
    group is enlarged to round(size * boost) rows with extra distinct
    lexicalizations. The base 2 * multiplier lexicalizations per frame are drawn
    first, from the same RNG stream as the ce-rb baseline, so val and test are
    identical for every (copy_attachments, boost). grammar_spec: which grammar's
    frames and lexicon to populate (GRAMMARS).
    """
    if boost < 1:
        raise ValueError(f"boost must be >= 1, got {boost}")
    rng = random.Random(seed)
    n_val = round(multiplier * val_frac)
    records = grammar_spec.records()
    lexicon = grammar_spec.lexicon
    base = [sample_lexicalizations(record["hi_frame"].split(), 2 * multiplier, rng, lexicon=lexicon) for record in records]
    rows = []
    for record, lexicalizations in zip(records, base):
        categories = record["hi_frame"].split()
        seen = set(lexicalizations)
        for task, chunk in (("copy", lexicalizations[:multiplier]), ("translate", lexicalizations[multiplier:])):
            keep_train = train_group_size(record, task, multiplier, n_val, copy_attachments) > 0
            for index, words in enumerate(chunk):
                split = split_for(record, task, index, multiplier, n_val)
                if split == "train" and not keep_train:
                    continue
                rows.append(make_row(record, task, split, words, grammar_spec))
        for task in ("copy", "translate"):
            size = train_group_size(record, task, multiplier, n_val, copy_attachments)
            n_extra = round(size * boost) - size
            if n_extra > 0:
                extra = sample_lexicalizations(categories, n_extra, rng, exclude=seen, lexicon=lexicon)
                seen.update(extra)
                rows.extend(make_row(record, task, "train", words, grammar_spec) for words in extra)
    return rows


def sequence_tokens(row):
    """Non-pad tokens of a row's training sequence: <bos> src <task> <sep> tgt <eos>."""
    return 2 * row["n_tokens"] + 4


def train_tokens(rows):
    """Train token budget: every non-pad token the model is trained on."""
    return sum(sequence_tokens(r) for r in rows if r["Split"] == "train")


def solve_boost(copy_attachments, multiplier=DEFAULT_MULTIPLIER, seed=DEFAULT_SEED, val_frac=DEFAULT_VAL_FRAC, grammar_spec=V1):
    """Uniform per-group boost factor that restores the grammar's ce-rb train
    token budget after dropping the copy rows of the other attachment sites."""
    baseline = train_tokens(build_examples(multiplier, seed, val_frac, grammar_spec=grammar_spec))
    kept = train_tokens(build_examples(multiplier, seed, val_frac, copy_attachments, grammar_spec=grammar_spec))
    return baseline / kept


def cerb_train_tokens(records, multiplier, val_frac=DEFAULT_VAL_FRAC):
    """Closed-form ce-rb train token budget of a frame inventory: every
    ambiguous frame contributes 2 * (multiplier - n_val) train rows (copy and
    translate), every unambiguous frame `multiplier` copy rows, each of
    2 * n_tokens + 4 tokens. Agrees with train_tokens(build_examples(...))."""
    n_val = round(multiplier * val_frac)
    return sum((2 * (multiplier - n_val) if r["is_ambiguous"] else multiplier) * (2 * r["n_tokens"] + 4) for r in records)


def solve_multiplier(target_tokens, val_frac=DEFAULT_VAL_FRAC, grammar_spec=V1):
    """The multiplier whose ce-rb train token budget is closest to target_tokens
    (the budget is non-decreasing in the multiplier)."""
    records = grammar_spec.records()
    multiplier = 1
    while cerb_train_tokens(records, multiplier + 1, val_frac) <= target_tokens:
        multiplier += 1
    return min((multiplier, multiplier + 1), key=lambda m: abs(cerb_train_tokens(records, m, val_frac) - target_tokens))


def train_breakdown(rows):
    """(task, attachment, gap) -> {"rows", "tokens", "frames"} over the train split."""
    out = {}
    for r in rows:
        if r["Split"] != "train":
            continue
        key = (r["Task"], r["Attachment Site"], r["Gap Type"])
        entry = out.setdefault(key, {"rows": 0, "tokens": 0, "frames": set()})
        entry["rows"] += 1
        entry["tokens"] += sequence_tokens(r)
        entry["frames"].add(r["HI Frame"])
    return out


def budget_table(variant, rows, boost, baseline_rows, reference=None):
    """Train rows / tokens / boost factor of the baseline and of this variant,
    preceded by an optional (label, rows) reference dataset of another
    grammar whose budget the baseline was matched to."""
    entries = [("ce-rb (baseline)", baseline_rows, 1.0)]
    if reference is not None:
        entries.insert(0, (*reference, 1.0))
    if variant != "ce-rb" or boost != 1.0:
        entries.append((variant, rows, boost))
    lines = ["| Dataset | Train rows | Train tokens | Boost factor |", "| --- | --- | --- | --- |"]
    for label, entry_rows, factor in entries:
        factor_text = "1" if factor == 1.0 else f"{factor:.4f}"
        lines.append(f"| {label} | {sum(r['Split'] == 'train' for r in entry_rows):,} | {train_tokens(entry_rows):,} | {factor_text} |")
    return lines


def breakdown_table(rows):
    """Train rows per (task, structure): frames, rows per frame, rows, tokens."""
    breakdown = train_breakdown(rows)
    lines = [
        "| Task | Attachment Site | Gap Type | Frames | Rows per frame | Rows | Tokens |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for task in ("translate", "copy"):
        for attachment, gap in CATEGORIES:
            entry = breakdown.get((task, attachment, gap))
            if entry is None:
                continue
            n_frames = len(entry["frames"])
            lines.append(
                f"| {task} | {attachment} | {gap} | {n_frames} | {entry['rows'] // n_frames} "
                f"| {entry['rows']:,} | {entry['tokens']:,} |"
            )
    total_rows = sum(e["rows"] for e in breakdown.values())
    total_tokens = sum(e["tokens"] for e in breakdown.values())
    lines.append(f"| **Total** | | | | | **{total_rows:,}** | **{total_tokens:,}** |")
    return lines


def has_agreement(grammar_spec=V1):
    """Whether the grammar's rows carry the number-agreement columns."""
    return {"Matrix Number", "RC Number"} <= {column for _, column in grammar_spec.extra_columns}


def markdown_row(cells):
    """`| a | b | |` with an empty cell rendered as `| |` (the other tables' total-row style)."""
    return "|" + "".join(f" {cell} |" if cell != "" else " |" for cell in cells)


def population_table(rows, grammar_spec=V1):
    """Per structure over every split and both tasks: frames (ambiguous /
    unambiguous), sentences (one per row: lexicalizations are distinct), the
    sample factor (lexicalizations drawn per frame; a `lo–hi` range if frames
    of the structure differ) and, for a grammar with number agreement, how
    many sentences have the RC verb agreeing with a noun of the same number
    as the matrix subject (`RC number = matrix number`) or of the other number
    (`RC number differs`); the no-RC structure shows `—` there."""
    agreement = has_agreement(grammar_spec)
    header = ["Attachment Site", "Gap Type", "Frames", "Ambiguous", "Unambiguous", "Sentences", "Sample factor"]
    if agreement:
        header += ["RC number = matrix number", "RC number differs"]
    lines = [markdown_row(header), markdown_row(["---"] * len(header))]
    totals = Counter()
    for attachment, gap in CATEGORIES:
        group = [r for r in rows if (r["Attachment Site"], r["Gap Type"]) == (attachment, gap)]
        if not group:
            continue
        frames = {r["HI Frame"]: r["is_ambiguous"] for r in group}
        n_ambiguous = sum(frames.values())
        per_frame = Counter(r["HI Frame"] for r in group).values()
        lo, hi = min(per_frame), max(per_frame)
        cells = [
            attachment,
            gap,
            len(frames),
            n_ambiguous,
            len(frames) - n_ambiguous,
            f"{len(group):,}",
            lo if lo == hi else f"{lo}–{hi}",
        ]
        totals.update(frames=len(frames), ambiguous=n_ambiguous, sentences=len(group))
        if agreement:
            with_rc = [r for r in group if r["RC Number"] != "None"]
            same = sum(r["RC Number"] == r["Matrix Number"] for r in with_rc)
            totals.update(same=same, differs=len(with_rc) - same)
            cells += [f"{same:,}", f"{len(with_rc) - same:,}"] if with_rc else ["—", "—"]
        lines.append(markdown_row(map(str, cells)))
    total_cells = [
        f"**{totals['frames']}**",
        f"**{totals['ambiguous']}**",
        f"**{totals['frames'] - totals['ambiguous']}**",
        f"**{totals['sentences']:,}**",
        "",
    ]
    if agreement:
        total_cells += [f"**{totals['same']:,}**", f"**{totals['differs']:,}**"]
    lines.append(markdown_row(["**Total**", "", *total_cells]))
    return lines


def stats_markdown(variant, copy_attachments, rows, boost, baseline_rows, grammar_spec=V1, reference=None):
    """Human-readable report of the variant: budget, per-structure train breakdown, split sizes."""
    counts = Counter(r["Split"] for r in rows)
    return "\n".join(
        [
            f"## `{grammar_spec.name}` / `{variant}`: {VARIANT_DESCRIPTIONS.get(variant, 'custom copy attachments')}",
            "",
            f"Train copy rows kept for attachment sites: {', '.join(copy_attachments)}. "
            f"Splits: train {counts['train']:,} / validation {counts['val']:,} / test {counts['test']:,} rows.",
            "",
            *budget_table(variant, rows, boost, baseline_rows, reference),
            "",
            *breakdown_table(rows),
        ]
    )


def dataset_card(rows, variant, copy_attachments, boost, multiplier, seed, baseline_rows, grammar_spec=V1, reference=None):
    records = grammar_spec.records()
    frames_repo_name = grammar_spec.frames_repo.split("/")[-1]
    train = [r for r in rows if r["Split"] == "train"]
    val = [r for r in rows if r["Split"] == "val"]
    test = [r for r in rows if r["Split"] == "test"]
    n_train_translate = sum(r["Task"] == "translate" for r in train)
    n_train_copy_ambiguous = sum(r["Task"] == "copy" and r["is_ambiguous"] for r in train)
    n_train_copy_unambiguous = sum(r["Task"] == "copy" and not r["is_ambiguous"] for r in train)
    n_val_translate = sum(r["Task"] == "translate" for r in val)
    upper = variant.upper()
    return "\n".join(
        [
            "---",
            # quoted: the ce-rb description contains ": ", which unquoted YAML reads as a mapping
            f"pretty_name: {json.dumps(f'{grammar_spec.title} {upper} ({VARIANT_DESCRIPTIONS[variant]})')}",
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
            f"# {grammar_spec.title} — {upper} Populated Frames",
            "",
            f"The {len(records)} depth-1 syntactic frames of "
            f"[{frames_repo_name}](https://huggingface.co/datasets/{grammar_spec.frames_repo}) "
            f"populated with randomly sampled lexicalizations, {multiplier} base lexicalizations per (frame, task) "
            f"(seeded RNG, seed={seed}; grammar source: `{grammar_spec.source}`).",
            "",
            *([grammar_spec.notes, ""] if grammar_spec.notes else []),
            *(
                [
                    f"The multiplier was solved so that the ce-rb train token budget is closest to the reference "
                    f"budget of {reference[0]} ({train_tokens(reference[1]):,} tokens).",
                    "",
                ]
                if reference is not None
                else []
            ),
            "Partition (README.md / DataGenFlow.png): ambiguous (in-distribution) frames contribute "
            "copy and translate examples split 90/10 into `train`/`validation` (per frame and task, "
            "so every ambiguous frame appears in both); unambiguous (out-of-distribution) frames "
            "contribute copy examples to `train` and translate examples to `test`. Copy and "
            "translate rows of a frame never share a sentence, so no test translate input appears "
            "as a train copy input.",
            "",
            f"## Copy-task variant `{variant}`: {VARIANT_DESCRIPTIONS[variant]}",
            "",
            "Train copy rows are kept only for frames whose relative clause attaches at one of these "
            f"sites: {', '.join(copy_attachments)} (`None` = no relative clause). The `validation` and "
            "`test` splits are identical across the `ce-rb`, `ce` and `rb` datasets (same base "
            "lexicalizations), so only the training data differs; `validation` therefore still contains "
            "copy rows of structures whose train copy rows were dropped. To keep the train token budget "
            "equal to `ce-rb`'s (non-pad tokens the model is trained on: `<bos> src <task> <sep> tgt <eos>` "
            "= 2 * n_tokens + 4 per row), every remaining train (frame, task) group is boosted by the same "
            "factor with additional distinct lexicalizations (rounded per group):",
            "",
            *budget_table(variant, rows, boost, baseline_rows, reference),
            "",
            f"- train: {len(train)} rows = {n_train_translate} translate (ambiguous) "
            f"+ {n_train_copy_ambiguous} copy (ambiguous) + {n_train_copy_unambiguous} copy (unambiguous)",
            f"- validation: {len(val)} rows = {n_val_translate} translate + {len(val) - n_val_translate} copy "
            "(ambiguous only)",
            f"- test: {len(test)} rows (translate, unambiguous)",
            "",
            "### Train rows per structure",
            "",
            *breakdown_table(rows),
            "",
            "`HF Sentence` is the task target: the HI sentence itself for `copy` rows, the "
            "hierarchical (Head Final) translation for `translate` rows. Frames use indexed "
            "bracketed notation where `[N_3]` means the terminal filled by HI token 3.",
            "",
            "## Frame counts per category (before population)",
            "",
            *counts_table(records),
            "",
            "## Sentence counts per category (after population)",
            "",
            "Over every split (`train`, `validation`, `test`) and both tasks. Every row is a distinct sentence, so "
            "`Sentences` counts rows and `Sample factor` is the number of lexicalizations drawn per frame (copy and "
            "translate together); it differs across structures wherever the variant dropped a structure's train copy "
            "rows (never replaced) or boosted the remaining train groups."
            + (
                " `RC number = matrix number` counts sentences whose RC verb agrees with a noun of the same number as "
                "the matrix subject, `RC number differs` those where the two numbers differ (`—` without an RC); a "
                "subject-gap RC on the subject always agrees, since its verb agrees with the matrix subject itself."
                if has_agreement(grammar_spec)
                else ""
            ),
            "",
            *population_table(rows, grammar_spec),
            "",
        ]
    )


def write_split_csvs(rows, out_dir, grammar_spec=V1):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split in ("train", "val", "test"):
        path = out_dir / f"{split}.csv"
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_columns(grammar_spec))
            writer.writeheader()
            writer.writerows(r for r in rows if r["Split"] == split)
        paths[f"data/{split}.csv"] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grammar", choices=tuple(GRAMMARS), default="v1", help="which grammar's frames to populate")
    parser.add_argument(
        "--variant", choices=tuple(COPY_VARIANTS), default=DEFAULT_VARIANT, help="which structures contribute train copy rows"
    )
    parser.add_argument(
        "--copy-attachments",
        nargs="+",
        choices=ATTACHMENT_SITES,
        default=None,
        help="override the variant's attachment sites (e.g. `Subject` alone drops the no-RC copy rows too)",
    )
    parser.add_argument(
        "--boost",
        type=float,
        default=None,
        help="per-group train boost factor (default: solved so the train token budget matches ce-rb)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=None,
        help=f"base examples per (frame, task) (default: {DEFAULT_MULTIPLIER} for v1, else solved from --train-tokens)",
    )
    parser.add_argument(
        "--train-tokens",
        type=float,
        default=None,
        help="ce-rb train token budget the multiplier is solved for (default: the v1 ce-rb budget; v1 itself only solves when given)",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-frac", type=float, default=DEFAULT_VAL_FRAC, help="ambiguous train/val carve")
    parser.add_argument(
        "--staging",
        type=Path,
        default=None,
        help="local staging dir for the split CSVs (default: ./data/<grammar repo prefix>-<variant>, per CLAUDE.md)",
    )
    parser.add_argument("--push", metavar="USER/REPO", help="push splits + dataset card to this HF dataset repo")
    args = parser.parse_args()

    grammar_spec = GRAMMARS[args.grammar]
    copy_attachments = tuple(args.copy_attachments or COPY_VARIANTS[args.variant])

    multiplier, reference = args.multiplier, None
    if grammar_spec is not V1 or args.train_tokens is not None:
        reference_rows = build_examples(DEFAULT_MULTIPLIER, args.seed, args.val_frac)
        reference = (f"Grammar V1 ce-rb, multiplier {DEFAULT_MULTIPLIER}", reference_rows)
        target = train_tokens(reference_rows) if args.train_tokens is None else args.train_tokens
        if multiplier is None:
            multiplier = solve_multiplier(target, args.val_frac, grammar_spec)
            budget = cerb_train_tokens(grammar_spec.records(), multiplier, args.val_frac)
            print(f"multiplier {multiplier}: ce-rb train budget {budget:,} tokens is closest to the target {target:,.0f}")
    if multiplier is None:
        multiplier = DEFAULT_MULTIPLIER

    boost = args.boost
    if boost is None:
        boost = solve_boost(copy_attachments, multiplier, args.seed, args.val_frac, grammar_spec)
    rows = build_examples(multiplier, args.seed, args.val_frac, copy_attachments, boost, grammar_spec)
    baseline_rows = build_examples(multiplier, args.seed, args.val_frac, grammar_spec=grammar_spec)

    staging = args.staging or PROJECT_ROOT / "data" / repo_name(args.variant, grammar_spec)
    paths = write_split_csvs(rows, staging, grammar_spec)
    card = dataset_card(rows, args.variant, copy_attachments, boost, multiplier, args.seed, baseline_rows, grammar_spec, reference)
    (staging / "README.md").write_text(card)
    for path in (*paths.values(), staging / "README.md"):
        print(f"wrote {path}")
    print()
    print(stats_markdown(args.variant, copy_attachments, rows, boost, baseline_rows, grammar_spec, reference))
    if args.push:
        print(f"pushed to {push_dataset(args.push, paths, card)}")


if __name__ == "__main__":
    main()
