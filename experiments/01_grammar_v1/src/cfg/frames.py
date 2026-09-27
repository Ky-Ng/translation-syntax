"""Depth-1 syntactic frame inventory: every HI frame with at most one relative clause.

First step of the dataset generation process (see DataGenFlow.png): enumerate
all frames the grammar generates with at most one RC (so no nesting, and no
RC on both subject and object), classify each by attachment site, gap type,
and ambiguity (linear reversal == hierarchical translation), then

- write the inventory report to results/frames_stats.md
- write the frame dataset (one copy row + one translate row per frame, with
  indexed bracketed frames) to artifacts/frames.csv
- optionally push the dataset to the HuggingFace Hub

Run with:
    cd experiments/01_grammar_v1/src && uv run python -m cfg.frames [--push USER/REPO]
"""

import argparse
import csv
from pathlib import Path

from nltk import Tree

from .grammar import enumerate_trees, hi_to_hf

# every single-RC frame fits in 8 tokens (N + 4-token RC + 3-token matrix VP);
# we enumerate a little past that and filter, rather than rely on the bound
MAX_LEN = 13

RC_LABELS = ("CP_subj_gap", "CP_obj_gap")

# (attachment site, gap type), in presentation order
CATEGORIES = [
    ("None", "None"),
    ("Subject", "Subject Gap"),
    ("Subject", "Object Gap"),
    ("Object", "Subject Gap"),
    ("Object", "Object Gap"),
]

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]

CSV_COLUMNS = ["Attachment Site", "Gap Type", "Task", "HI Frame", "HF Frame", "is_ambiguous", "n_tokens", "depth"]


def _indexed(tree, counter):
    """Copy a frame tree, wrapping each terminal-category leaf into a node
    holding its 1-based HI linear position, so the HF mirror keeps HI indices."""
    if isinstance(tree, str):
        counter[0] += 1
        return Tree(tree, [counter[0]])
    return Tree(tree.label(), [_indexed(child, counter) for child in tree])


def bracketed(tree):
    """Render an indexed frame tree as e.g. [S [NP [N_1]] [VP [VP_intrans [V_intrans_2]]]].

    A terminal is a node whose single child is its HI index (see _indexed), so
    this works for any grammar's terminal categories."""
    if len(tree) == 1 and isinstance(tree[0], int):
        return f"[{tree.label()}_{tree[0]}]"
    return "[" + tree.label() + " " + " ".join(bracketed(child) for child in tree) + "]"


def single_rc_records():
    """All HI frames with at most one RC, as dicts sorted by category.

    Keys: attachment, gap, hi_frame, hf_frame (flat category strings),
    hi_bracketed, hf_bracketed (indexed bracketed trees), is_ambiguous,
    n_tokens, depth.
    """
    records = []
    for frame, tree in enumerate_trees("hi", MAX_LEN):
        rc_nodes = [t for t in tree.subtrees() if t.label() in RC_LABELS]
        if len(rc_nodes) > 1:
            continue
        if not rc_nodes:
            attachment, gap = "None", "None"
        else:
            subject_np = tree[0]  # S -> NP VP
            in_subject = any(t is rc_nodes[0] for t in subject_np.subtrees())
            attachment = "Subject" if in_subject else "Object"
            gap = "Subject Gap" if rc_nodes[0].label() == "CP_subj_gap" else "Object Gap"
        hf_frame = hi_to_hf(tree).leaves()
        linear = frame[:1] + frame[:0:-1]
        hi_indexed = _indexed(tree, [0])
        hf_indexed = hi_to_hf(hi_indexed)
        records.append(
            {
                "attachment": attachment,
                "gap": gap,
                "hi_frame": " ".join(frame),
                "hf_frame": " ".join(hf_frame),
                "hi_bracketed": bracketed(hi_indexed),
                "hf_bracketed": bracketed(hf_indexed),
                # 1-based HI positions in HF linear order: the token permutation
                # that translates a populated HI sentence into its HF sentence
                "hf_order": tuple(hf_indexed.leaves()),
                "is_ambiguous": linear == hf_frame,
                "n_tokens": len(frame),
                "depth": len(rc_nodes),
            }
        )
    order = {category: i for i, category in enumerate(CATEGORIES)}
    records.sort(key=lambda r: (order[(r["attachment"], r["gap"])], r["n_tokens"], r["hi_frame"]))
    return records


def task_rows(records):
    """Expand each frame into a copy row (target = the HI frame itself) and a
    translate row (target = the mirrored HF frame)."""
    rows = []
    for r in records:
        for task in ("copy", "translate"):
            rows.append(
                {
                    "Attachment Site": r["attachment"],
                    "Gap Type": r["gap"],
                    "Task": task,
                    "HI Frame": r["hi_bracketed"],
                    "HF Frame": r["hi_bracketed"] if task == "copy" else r["hf_bracketed"],
                    "is_ambiguous": r["is_ambiguous"],
                    "n_tokens": r["n_tokens"],
                    "depth": r["depth"],
                }
            )
    return rows


def counts_table(records):
    lines = [
        "| Attachment Site | Gap Type | Frames | Ambiguous | Unambiguous |",
        "| --- | --- | --- | --- | --- |",
    ]
    for category in CATEGORIES:
        rows = [r for r in records if (r["attachment"], r["gap"]) == category]
        n_ambiguous = sum(r["is_ambiguous"] for r in rows)
        lines.append(f"| {category[0]} | {category[1]} | {len(rows)} | {n_ambiguous} | {len(rows) - n_ambiguous} |")
    n_ambiguous = sum(r["is_ambiguous"] for r in records)
    lines.append(f"| **Total** | | **{len(records)}** | **{n_ambiguous}** | **{len(records) - n_ambiguous}** |")
    return lines


def frames_stats_markdown(records):
    ambiguous = [r for r in records if r["is_ambiguous"]]
    lines = [
        "Generated by [src/cfg/frames.py](../src/cfg/frames.py) from the grammar in "
        "[src/cfg/grammar.py](../src/cfg/grammar.py) — do not edit by hand "
        "(`cd src && uv run python -m cfg.frames`).",
        "",
        "# Depth-1 Syntactic Frames (at most one RC)",
        "",
        f"**{len(records)} frames total: {len(ambiguous)} ambiguous (in distribution), "
        f"{len(records) - len(ambiguous)} unambiguous (out of distribution).**",
        "",
        "A frame is ambiguous iff the linear reversal rule (keep the first token, reverse "
        "the rest) produces the same HF string as the hierarchical translation.",
        "",
        "## Counts per Category",
        "",
        *counts_table(records),
        "",
        "## Frame Inventory",
        "",
        "| Attachment Site | Gap Type | HI Frame | HF Frame | is_ambiguous | n_tokens | depth |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(
            f"| {r['attachment']} | {r['gap']} | {r['hi_frame']} | {r['hf_frame']} "
            f"| {r['is_ambiguous']} | {r['n_tokens']} | {r['depth']} |"
        )
    return "\n".join(lines) + "\n"


def dataset_card(records):
    n_ambiguous = sum(r["is_ambiguous"] for r in records)
    return "\n".join(
        [
            "---",
            "pretty_name: Grammar V1 Depth-1 Syntactic Frames",
            "---",
            "",
            "# Grammar V1 — Depth-1 Syntactic Frames",
            "",
            "All syntactic frames with at most one relative clause generated by the paired "
            "Head Initial / Head Final CFG of the translation-syntax project "
            "(`experiments/01_grammar_v1/src/cfg/grammar.py`). Frames are category-level "
            "skeletons to be populated with lexical items; terminals carry their 1-based "
            "HI linear position (e.g. `[N_1]`), so HF frames show where each HI slot moved.",
            "",
            f"{len(records)} frames ({n_ambiguous} ambiguous / in distribution, "
            f"{len(records) - n_ambiguous} unambiguous / out of distribution), each expanded "
            "into a `copy` row (HF Frame = HI Frame, the copy target) and a `translate` row "
            "(HF Frame = hierarchical translation), giving "
            f"{2 * len(records)} rows in `frames.csv`.",
            "",
            "`is_ambiguous` is True iff the linear reversal rule (keep the first token, "
            "reverse the rest) matches the hierarchical translation.",
            "",
            "## Counts per Category (frames, before task expansion)",
            "",
            *counts_table(records),
            "",
        ]
    )


def push_to_hub(repo_id, csv_path, records):
    from .hub import push_dataset

    return push_dataset(repo_id, {"frames.csv": csv_path}, dataset_card(records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "frames_stats.md")
    parser.add_argument("--csv", type=Path, default=EXPERIMENT_DIR / "artifacts" / "frames.csv")
    parser.add_argument("--push", metavar="USER/REPO", help="push frames.csv + dataset card to this HF dataset repo")
    args = parser.parse_args()

    records = single_rc_records()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(frames_stats_markdown(records))
    print(f"wrote {args.out}")

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(task_rows(records))
    print(f"wrote {args.csv}")

    if args.push:
        print(f"pushed to {push_to_hub(args.push, args.csv, records)}")


if __name__ == "__main__":
    main()
