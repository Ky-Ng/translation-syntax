"""Depth-1 syntactic frame inventory of the number-agreement grammar.

Counterpart of frames.py for cfg.grammar_number_agreement: enumerate every HI
frame with at most one relative clause, classify it by attachment site, gap
type and ambiguity exactly as frames.py does, and add the agreement features

- Matrix Number: number of the matrix subject (and so of the matrix verb)
- RC Number: number of the noun the RC verb agrees with (the head noun for a
  subject gap, the embedded subject for an object gap); None without an RC
- n_attractors: nouns between a verb and the noun it agrees with that carry
  the other number (agreement attractors). Only center-embedded frames can
  have one, which is the pressure this grammar adds over Grammar V1.

Every V1 frame appears in 2^(number of nouns) number variants. Outputs:

- the inventory report, results/frames_number_agreement_stats.md
- the frame dataset (one copy row + one translate row per frame, indexed
  bracketed frames), artifacts/frames_number_agreement.csv
- optionally the dataset pushed to the HuggingFace Hub (--push)

Run with:
    cd experiments/01_grammar_v1/src && uv run python -m cfg.frames_number_agreement [--push USER/REPO]
"""

import argparse
import csv
from pathlib import Path

from . import grammar_number_agreement as g
from .frames import CATEGORIES, EXPERIMENT_DIR, MAX_LEN, RC_LABELS, _indexed, bracketed, counts_table
from .hub import push_dataset

CSV_COLUMNS = [
    "Attachment Site",
    "Gap Type",
    "Matrix Number",
    "RC Number",
    "n_attractors",
    "Task",
    "HI Frame",
    "HF Frame",
    "is_ambiguous",
    "n_tokens",
    "depth",
]

# stock words for the human-readable example of each frame: the i-th noun /
# verb slot of a frame takes the i-th lemma, inflected to the slot's number
EXAMPLE_LEMMAS = {"N": ("cat", "dog", "bird"), "V_trans": ("pursue", "hug"), "V_intrans": ("swim", "sleep")}
EXAMPLE_WORDS = {"Adv": "quickly", "C": "that"}


def example_words(frame):
    """A deterministic lexicalization of a frame's category tokens."""
    used = {category: 0 for category in EXAMPLE_LEMMAS}
    words = []
    for token in frame:
        category = g.strip_number(token)
        if category in EXAMPLE_WORDS:
            words.append(EXAMPLE_WORDS[category])
        else:
            lemma = EXAMPLE_LEMMAS[category][used[category] % len(EXAMPLE_LEMMAS[category])]
            used[category] += 1
            words.append(g.inflect(category, lemma, g.number_of(token)))
    return words


def _number_key(frame):
    return tuple(-1 if g.number_of(token) is None else g.NUMBERS.index(g.number_of(token)) for token in frame)


def single_rc_records():
    """All HI frames with at most one RC, as dicts sorted by category, then by
    V1 frame, then singular before plural.

    Keys: those of frames.single_rc_records plus v1_frame (the frame with
    number features erased), matrix_number, rc_number, n_attractors,
    hi_example, hf_example.
    """
    records = []
    for frame, tree in g.enumerate_trees("hi", MAX_LEN):
        rc_nodes = [t for t in tree.subtrees() if g.strip_number(t.label()) in RC_LABELS]
        if len(rc_nodes) > 1:
            continue
        if not rc_nodes:
            attachment, gap = "None", "None"
        else:
            subject_np = tree[0]  # S -> NP_n VP_n
            in_subject = any(t is rc_nodes[0] for t in subject_np.subtrees())
            attachment = "Subject" if in_subject else "Object"
            gap = "Subject Gap" if g.strip_number(rc_nodes[0].label()) == "CP_subj_gap" else "Object Gap"
        hf_frame = g.hi_to_hf(tree).leaves()
        linear = frame[:1] + frame[:0:-1]
        hi_indexed = _indexed(tree, [0])
        hf_indexed = g.hi_to_hf(hi_indexed)
        hf_order = tuple(hf_indexed.leaves())
        dependencies = g.agreement_dependencies(tree)
        matrix_noun = next(noun for noun, _, kind in dependencies if kind == "matrix")
        rc_nouns = [noun for noun, _, kind in dependencies if kind != "matrix"]
        hi_words = example_words(frame)
        records.append(
            {
                "attachment": attachment,
                "gap": gap,
                "matrix_number": g.number_of(frame[matrix_noun]),
                "rc_number": g.number_of(frame[rc_nouns[0]]) if rc_nouns else "None",
                "n_attractors": g.n_attractors(tree),
                "v1_frame": " ".join(g.strip_number(token) for token in frame),
                "hi_frame": " ".join(frame),
                "hf_frame": " ".join(hf_frame),
                "hi_bracketed": bracketed(hi_indexed),
                "hf_bracketed": bracketed(hf_indexed),
                "hf_order": hf_order,
                "hi_example": " ".join(hi_words),
                "hf_example": " ".join(hi_words[position - 1] for position in hf_order),
                "is_ambiguous": linear == hf_frame,
                "n_tokens": len(frame),
                "depth": len(rc_nodes),
            }
        )
    order = {category: i for i, category in enumerate(CATEGORIES)}
    records.sort(
        key=lambda r: (order[(r["attachment"], r["gap"])], r["n_tokens"], r["v1_frame"], _number_key(r["hi_frame"].split()))
    )
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
                    "Matrix Number": r["matrix_number"],
                    "RC Number": r["rc_number"],
                    "n_attractors": r["n_attractors"],
                    "Task": task,
                    "HI Frame": r["hi_bracketed"],
                    "HF Frame": r["hi_bracketed"] if task == "copy" else r["hf_bracketed"],
                    "is_ambiguous": r["is_ambiguous"],
                    "n_tokens": r["n_tokens"],
                    "depth": r["depth"],
                }
            )
    return rows


def agreement_table(records):
    """Per category: frames, whether the RC verb agrees with the same number as
    the matrix verb, and frames with an agreement attractor."""
    lines = [
        "| Attachment Site | Gap Type | Frames | RC number = matrix number | RC number differs | With attractor |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for category in CATEGORIES:
        rows = [r for r in records if (r["attachment"], r["gap"]) == category]
        with_rc = [r for r in rows if r["rc_number"] != "None"]
        same = sum(r["rc_number"] == r["matrix_number"] for r in with_rc)
        attractor = sum(r["n_attractors"] > 0 for r in rows)
        lines.append(f"| {category[0]} | {category[1]} | {len(rows)} | {same} | {len(with_rc) - same} | {attractor} |")
    n_attractor = sum(r["n_attractors"] > 0 for r in records)
    lines.append(f"| **Total** | | **{len(records)}** | | | **{n_attractor}** |")
    return lines


def frames_stats_markdown(records):
    ambiguous = [r for r in records if r["is_ambiguous"]]
    n_v1 = len({r["v1_frame"] for r in records})
    lines = [
        "Generated by [src/cfg/frames_number_agreement.py](../src/cfg/frames_number_agreement.py) from the "
        "grammar in [src/cfg/grammar_number_agreement.py](../src/cfg/grammar_number_agreement.py) — do not "
        "edit by hand (`cd src && uv run python -m cfg.frames_number_agreement`).",
        "",
        "# Depth-1 Syntactic Frames with Number Agreement (at most one RC)",
        "",
        f"**{len(records)} frames total: {len(ambiguous)} ambiguous (in distribution), "
        f"{len(records) - len(ambiguous)} unambiguous (out of distribution).** "
        f"Each of the {n_v1} Grammar V1 frames appears in 2^(number of nouns) number variants "
        "(`_sg` / `_pl` on N, V_intrans and V_trans); verbs agree with their subject, objects are free.",
        "",
        "A frame is ambiguous iff the linear reversal rule (keep the first token, reverse "
        "the rest) produces the same HF string as the hierarchical translation. Number features "
        "never change this: they only reorder like any other token.",
        "",
        "Matrix Number is the number of the matrix subject and verb; RC Number that of the noun the RC verb "
        "agrees with (head noun for a subject gap, embedded subject for an object gap). An attractor is a noun "
        "of the other number between a verb and the noun it agrees with, which only center embedding produces.",
        "",
        "## Counts per Category",
        "",
        *counts_table(records),
        "",
        "## Agreement per Category",
        "",
        *agreement_table(records),
        "",
        "## Frame Inventory",
        "",
        "| Attachment Site | Gap Type | Matrix | RC | Attractors | HI Frame | HF Frame | HI Example | HF Example "
        "| is_ambiguous | n_tokens | depth |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in records:
        lines.append(
            f"| {r['attachment']} | {r['gap']} | {r['matrix_number']} | {r['rc_number']} | {r['n_attractors']} "
            f"| {r['hi_frame']} | {r['hf_frame']} | {r['hi_example']} | {r['hf_example']} "
            f"| {r['is_ambiguous']} | {r['n_tokens']} | {r['depth']} |"
        )
    return "\n".join(lines) + "\n"


def dataset_card(records):
    n_ambiguous = sum(r["is_ambiguous"] for r in records)
    n_v1 = len({r["v1_frame"] for r in records})
    return "\n".join(
        [
            "---",
            "pretty_name: Grammar V1 + Number Agreement Depth-1 Syntactic Frames",
            "---",
            "",
            "# Grammar V1 + Number Agreement — Depth-1 Syntactic Frames",
            "",
            "All syntactic frames with at most one relative clause generated by the paired "
            "Head Initial / Head Final CFG with subject-verb number agreement of the translation-syntax "
            "project (`experiments/01_grammar_v1/src/cfg/grammar_number_agreement.py`). The grammar is "
            "[Grammar V1](https://huggingface.co/datasets/kylelovesllms/grammar-v1-frames) with a number "
            "feature (`_sg` / `_pl`) on nouns, verbs and the phrases that carry their number: the matrix verb "
            "agrees with the matrix subject, a subject-gap RC verb with the head noun, an object-gap RC verb "
            "with the embedded subject; objects take either number. Frames are category-level skeletons to "
            "be populated with lexical items; terminals carry their 1-based HI linear position "
            "(e.g. `[N_pl_1]`), so HF frames show where each HI slot moved.",
            "",
            f"{len(records)} frames ({n_ambiguous} ambiguous / in distribution, {len(records) - n_ambiguous} "
            f"unambiguous / out of distribution; each of the {n_v1} V1 frames in 2^(number of nouns) number "
            "variants), each expanded into a `copy` row (HF Frame = HI Frame, the copy target) and a "
            f"`translate` row (HF Frame = hierarchical translation), giving {2 * len(records)} rows in `frames.csv`.",
            "",
            "`is_ambiguous` is True iff the linear reversal rule (keep the first token, reverse the rest) "
            "matches the hierarchical translation. `Matrix Number` is the number of the matrix subject, "
            "`RC Number` that of the noun the RC verb agrees with (`None` without an RC), and `n_attractors` "
            "counts nouns of the other number between a verb and the noun it agrees with (only center-embedded "
            "frames have any).",
            "",
            "## Counts per Category (frames, before task expansion)",
            "",
            *counts_table(records),
            "",
            "## Agreement per Category",
            "",
            *agreement_table(records),
            "",
        ]
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "results" / "frames_number_agreement_stats.md")
    parser.add_argument("--csv", type=Path, default=EXPERIMENT_DIR / "artifacts" / "frames_number_agreement.csv")
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
        print(f"pushed to {push_dataset(args.push, {'frames.csv': args.csv}, dataset_card(records))}")


if __name__ == "__main__":
    main()
