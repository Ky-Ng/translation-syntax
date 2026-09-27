"""Checkpoint evaluator for the ce-rb runs: exact match, token accuracy and the
mispredicted tokens on the validation and test splits.

Per example (content at position shift 0 unless `shift` is given):
  * teacher-forced token accuracy over the target tokens (incl. <eos>): given
    the gold prefix, is the argmax next token the gold token?  Every mismatch is
    recorded as ``<1-based target position>:<gold>-><pred>``.
  * greedy exact match: decode from ``<bos> src <task> <sep>`` until <eos> and
    compare with the gold target.  This equals "every teacher-forced position is
    correct" by construction; the greedy string additionally shows what the
    model actually emits once it diverges.
  * rule: ``hierarchical`` if greedy == gold, ``linear`` if greedy == the
    linear-reversal target (README: keep the first word, reverse the rest),
    else ``other``.  Copy rows are ``copy`` / ``other``.

Library use (training.train calls this at every checkpoint):
    examples = {name: prepare_examples(splits[name]) for name in ("validation", "test")}
    report = run_eval(model, examples, device, out_dir=run_dir / "eval" / "step_002000", step=2000)
    wandb.log(report.metrics, step=2000)

Post hoc over saved checkpoints (from experiments/01_grammar_v1/src):
    uv run python -m training.evaluate --runs "gpt2-8h2l300Ks4lr-ce-rb-seed42" --steps last
    uv run python -m training.evaluate --runs "gpt2-8h*" --skip-existing

Outputs under artifacts/<run>/eval/step_XXXXXX/: <split>.csv (one row per
example, with its mispredicted tokens), confusions_<split>.csv (gold->pred
counts over the mispredictions), summary.csv (per split x task x category)
and errors_<split>.csv (per category: rule mix, error types, dominant output
orderings as source positions, error positions, top confusions; long format
with a `kind` column).  artifacts/<run>/eval/summary.csv and errors.csv
concatenate every evaluated step.  `--from-csv` rebuilds errors_*.csv from
existing per-example CSVs without re-running any model.
"""

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

from training.data import (
    CONTEXT_LEN,
    DEFAULT_DATASET,
    PAD_ID,
    STOI,
    VOCAB,
    encode_row,
    grammar_for,
    load_splits,
    select_grammar,
)

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = EXPERIMENT_DIR / "artifacts"
EOS_ID = STOI["<eos>"]
SEP_ID = STOI["<sep>"]
DEFAULT_SPLITS = ("validation", "test")
MAX_NEW_TOKENS = 12  # longest target is 8 words + <eos>; the slack exposes run-on outputs

EXAMPLE_COLUMNS = [
    "step", "split", "task", "attachment_site", "gap_type", "is_ambiguous", "hi_frame",
    "source", "target", "greedy", "exact_match", "token_acc", "n_target", "n_wrong",
    "mispredictions", "linear_target", "rule", "eos_emitted",
]
SUMMARY_COLUMNS = [
    "step", "split", "task", "attachment_site", "gap_type", "n",
    "exact_match", "token_acc", "linear_rate", "other_rate",
]
ERROR_COLUMNS = ["step", "split", "task", "attachment_site", "gap_type", "n", "kind", "key", "count", "share"]
ERROR_KINDS = ("rule", "error_type", "ordering", "error_position", "first_error_position", "confusion")


def linear_reversal(sentence):
    """README "Linear Reversal Rule": keep the first word, reverse the rest."""
    words = sentence.split()
    return " ".join(words[:1] + words[:0:-1])


@dataclass
class Example:
    task: str
    source: str
    target: str
    prefix_ids: list  # <bos> src <task> <sep>
    target_ids: list  # tgt <eos>
    meta: dict = field(default_factory=dict)


def _label(value):
    """Category label; the csv loader reads the literal "None" (no RC) as null."""
    return "None" if value is None or value == "" else str(value)


def prepare_examples(split):
    """Encode a HF split (columns from cfg.populate) into prefix/target ids."""
    examples = []
    for row in split:
        task, source, target = row["Task"], row["HI Sentence"], row["HF Sentence"]
        ids = encode_row(task, source, target)
        cut = ids.index(SEP_ID) + 1
        meta = {
            "attachment_site": _label(row.get("Attachment Site")),
            "gap_type": _label(row.get("Gap Type")),
            "is_ambiguous": row.get("is_ambiguous", ""),
            "hi_frame": row.get("HI Frame", ""),
        }
        examples.append(Example(task, source, target, ids[:cut], ids[cut:], meta))
    return examples


def _batches(items, size):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _pad_batch(sequences, shift):
    """Right-padded [B, CONTEXT_LEN] input_ids / attention_mask with each
    sequence occupying positions [shift, shift + len)."""
    input_ids = torch.full((len(sequences), CONTEXT_LEN), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), CONTEXT_LEN), dtype=torch.long)
    for i, seq in enumerate(sequences):
        input_ids[i, shift : shift + len(seq)] = torch.tensor(seq, dtype=torch.long)
        attention_mask[i, shift : shift + len(seq)] = 1
    return input_ids, attention_mask


@torch.inference_mode()
def teacher_forced_predictions(model, examples, device, batch_size=512, shift=0):
    """Argmax next-token prediction at every target position given the gold prefix."""
    predictions = []
    for batch in _batches(examples, batch_size):
        input_ids, attention_mask = _pad_batch([ex.prefix_ids + ex.target_ids for ex in batch], shift)
        logits = model(input_ids=input_ids.to(device), attention_mask=attention_mask.to(device)).logits
        argmax = logits.argmax(-1).cpu()
        for i, ex in enumerate(batch):
            first = shift + len(ex.prefix_ids) - 1  # logits here predict the first target token
            predictions.append(argmax[i, first : first + len(ex.target_ids)].tolist())
    return predictions


@torch.inference_mode()
def greedy_decode(model, examples, device, batch_size=512, shift=0, max_new_tokens=MAX_NEW_TOKENS):
    """Greedy continuation of each prefix until <eos> (included) or max_new_tokens."""
    outputs = []
    for batch in _batches(examples, batch_size):
        input_ids, attention_mask = _pad_batch([ex.prefix_ids for ex in batch], shift)
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        rows = torch.arange(len(batch), device=device)
        cursor = torch.tensor([shift + len(ex.prefix_ids) for ex in batch], device=device)
        finished = torch.zeros(len(batch), dtype=torch.bool, device=device)
        history = []
        for _ in range(max_new_tokens):
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            next_ids = logits[rows, cursor - 1].argmax(-1)
            active = ~finished
            input_ids[rows[active], cursor[active]] = next_ids[active]
            attention_mask[rows[active], cursor[active]] = 1
            history.append((next_ids.cpu(), active.cpu()))
            cursor = cursor + active.long()
            finished = finished | (next_ids == EOS_ID)
            if bool(finished.all()):
                break
        for i in range(len(batch)):
            outputs.append([int(next_ids[i]) for next_ids, active in history if active[i]])
    return outputs


def _words(ids):
    return " ".join(VOCAB[i] for i in ids)


def score_example(example, tf_pred, greedy, split, step):
    """One per-example result row (see EXAMPLE_COLUMNS)."""
    gold = example.target_ids
    mismatches = [(pos, g, p) for pos, (g, p) in enumerate(zip(gold, tf_pred), start=1) if g != p]
    eos_emitted = bool(greedy) and greedy[-1] == EOS_ID
    greedy_text = _words(greedy[:-1] if eos_emitted else greedy)
    exact_match = greedy == gold
    if example.task == "translate":
        linear_target = linear_reversal(example.source)
        rule = "hierarchical" if exact_match else "linear" if greedy_text == linear_target else "other"
    else:
        linear_target = ""
        rule = "copy" if exact_match else "other"
    return {
        "step": step,
        "split": split,
        "task": example.task,
        "attachment_site": example.meta.get("attachment_site", ""),
        "gap_type": example.meta.get("gap_type", ""),
        "is_ambiguous": example.meta.get("is_ambiguous", ""),
        "hi_frame": example.meta.get("hi_frame", ""),
        "source": example.source,
        "target": example.target,
        "greedy": greedy_text,
        "exact_match": exact_match,
        "token_acc": (len(gold) - len(mismatches)) / len(gold),
        "n_target": len(gold),
        "n_wrong": len(mismatches),
        "mispredictions": ";".join(f"{pos}:{VOCAB[g]}->{VOCAB[p]}" for pos, g, p in mismatches),
        "linear_target": linear_target,
        "rule": rule,
        "eos_emitted": eos_emitted,
    }


def summarize(rows, split, step):
    """Aggregate example rows per (task, attachment site, gap type), plus
    per-task and overall rows marked with "all"."""
    groups = defaultdict(list)
    for row in rows:
        groups[("all", "all", "all")].append(row)
        groups[(row["task"], "all", "all")].append(row)
        groups[(row["task"], row["attachment_site"], row["gap_type"])].append(row)
    summary = []
    for (task, site, gap), members in sorted(groups.items(), key=lambda kv: [str(k) for k in kv[0]]):
        n_tokens = sum(r["n_target"] for r in members)
        summary.append({
            "step": step,
            "split": split,
            "task": task,
            "attachment_site": site,
            "gap_type": gap,
            "n": len(members),
            "exact_match": sum(r["exact_match"] for r in members) / len(members),
            "token_acc": sum(r["n_target"] - r["n_wrong"] for r in members) / n_tokens,
            "linear_rate": sum(r["rule"] == "linear" for r in members) / len(members),
            "other_rate": sum(r["rule"] == "other" for r in members) / len(members),
        })
    return summary


def wandb_metrics(summary):
    """Flat {"<split>[/<task>[/<Site>_<Gap>]]/<metric>": value} for wandb.log."""
    metrics = {}
    for row in summary:
        parts = [row["split"]]
        if row["task"] != "all":
            parts.append(row["task"])
        if row["attachment_site"] != "all":
            parts.append(f'{row["attachment_site"]}_{row["gap_type"]}'.replace(" ", ""))
        key = "/".join(parts)
        metrics[f"{key}/exact_match"] = row["exact_match"]
        metrics[f"{key}/token_acc"] = row["token_acc"]
        if row["task"] == "translate":
            metrics[f"{key}/linear_rate"] = row["linear_rate"]
    return metrics


def confusions(rows):
    """Counter of (gold word, predicted word) over all teacher-forced mispredictions."""
    counter = Counter()
    for row in rows:
        for item in filter(None, row["mispredictions"].split(";")):
            gold, pred = item.split(":", 1)[1].split("->")
            counter[(gold, pred)] += 1
    return counter


def source_positions(source, words):
    """Each word as its 1-based position in `source` ('?' if absent; first occurrence)."""
    index = {}
    for i, word in enumerate(source.split(), start=1):
        index.setdefault(word, i)
    return " ".join(str(index.get(word, "?")) for word in words)


def error_type(row):
    """Coarse label for a wrong output: linear | permutation (gold words, wrong
    order) | flags among missing / repeat / foreign (+ no-eos), joined by '+'."""
    if row["exact_match"]:
        return "correct"
    if row["rule"] == "linear":
        return "linear"
    out, gold = row["greedy"].split(), row["target"].split()
    flags = []
    if sorted(out) == sorted(gold):
        flags.append("permutation")
    else:
        out_counts, gold_counts = Counter(out), Counter(gold)
        if any(gold_counts[w] > out_counts[w] for w in gold_counts):
            flags.append("missing")
        if any(out_counts[w] > gold_counts[w] for w in out_counts if w in gold_counts):
            flags.append("repeat")
        if any(w not in gold_counts for w in out_counts):
            flags.append("foreign")
    if not row["eos_emitted"]:
        flags.append("no-eos")
    return "+".join(flags)


def error_breakdown(rows, split, step, top=8):
    """Long-format per-category error analysis (ERROR_COLUMNS rows)."""
    out = []
    groups = defaultdict(list)
    for row in rows:
        groups[(row["task"], row["attachment_site"], row["gap_type"])].append(row)
    for (task, site, gap), members in sorted(groups.items()):
        n = len(members)
        base = {"step": step, "split": split, "task": task, "attachment_site": site, "gap_type": gap, "n": n}

        def emit(kind, counter, limit=None, base=base, n=n):
            items = counter.most_common(limit) if limit else sorted(counter.items(), key=lambda kv: (len(kv[0]), kv[0]))
            for key, count in items:
                out.append({**base, "kind": kind, "key": key, "count": count, "share": count / n})

        wrong = [r for r in members if not r["exact_match"]]
        emit("rule", Counter(r["rule"] for r in members))
        emit("error_type", Counter(error_type(r) for r in wrong))
        emit("ordering", Counter(
            f"gold {source_positions(r['source'], r['target'].split())} -> model {source_positions(r['source'], r['greedy'].split())}"
            for r in wrong), top)
        positions, first, confusion = Counter(), Counter(), Counter()
        for r in wrong:
            items = [m for m in r["mispredictions"].split(";") if m]
            found = [int(m.split(":", 1)[0]) for m in items]
            for pos in found:
                positions[str(pos)] += 1
            if found:
                first[str(min(found))] += 1
            for m in items:
                confusion[m.split(":", 1)[1]] += 1
        emit("error_position", positions)
        emit("first_error_position", first)
        emit("confusion", confusion, top)
    return out


def format_error_breakdown(errors, splits=("test",), top=3):
    """Human-readable per-category digest of error_breakdown() rows."""
    lines = []
    keys = sorted({(r["split"], r["task"], r["attachment_site"], r["gap_type"]) for r in errors if r["split"] in splits})
    for split, task, site, gap in keys:
        rows = [r for r in errors if (r["split"], r["task"], r["attachment_site"], r["gap_type"]) == (split, task, site, gap)]
        n = rows[0]["n"]

        def kind(name, limit=None, rows=rows):
            items = [r for r in rows if r["kind"] == name]
            items = sorted(items, key=lambda r: -r["count"])[:limit] if limit else items
            return ", ".join(f"{r['key']} {r['share']:.2f}" for r in items) or "-"

        lines.append(f"  [{split}] {task} {site} / {gap}  n={n}")
        lines.append(f"     rule        : {kind('rule')}")
        lines.append(f"     error types : {kind('error_type', top + 2)}")
        lines.append(f"     first error : {kind('first_error_position')}")
        for r in sorted([r for r in rows if r["kind"] == "ordering"], key=lambda r: -r["count"])[:top]:
            lines.append(f"     ordering    : {r['key']}  x{r['count']} ({r['share']:.2f})")
        lines.append(f"     confusions  : {kind('confusion', top + 2)}")
    return "\n".join(lines)


def read_example_csv(path):
    """Per-example rows back from <split>.csv with the typed columns restored."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["exact_match"] = row["exact_match"] == "True"
        row["eos_emitted"] = row["eos_emitted"] == "True"
        row["n_target"], row["n_wrong"] = int(row["n_target"]), int(row["n_wrong"])
        row["token_acc"] = float(row["token_acc"])
        row["step"] = int(row["step"]) if row["step"] else None
    return rows


def format_summary_line(summary):
    """One-line digest: ``validation: EM 0.983 tok 0.997 | test: EM 0.120 tok 0.801 linear 0.750``."""
    parts = []
    for split in dict.fromkeys(r["split"] for r in summary):
        overall = next(r for r in summary if r["split"] == split and r["task"] == "all")
        text = f"{split}: EM {overall['exact_match']:.3f} tok {overall['token_acc']:.3f}"
        translate = next((r for r in summary if r["split"] == split and r["task"] == "translate" and r["attachment_site"] == "all"), None)
        if translate:
            text += f" linear {translate['linear_rate']:.3f}"
        parts.append(text)
    return " | ".join(parts)


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


@dataclass
class EvalReport:
    metrics: dict  # flat, wandb-ready
    summary: list  # SUMMARY_COLUMNS rows for every split
    rows: dict  # split -> EXAMPLE_COLUMNS rows
    errors: dict = field(default_factory=dict)  # split -> ERROR_COLUMNS rows


def run_eval(model, examples_by_split, device, out_dir=None, step=None, batch_size=512, shift=0):
    """Evaluate `model` on {split: [Example]}; optionally write the CSVs to out_dir."""
    was_training = model.training
    model.eval()
    report = EvalReport(metrics={}, summary=[], rows={})
    try:
        for split, examples in examples_by_split.items():
            if not examples:
                continue
            tf_preds = teacher_forced_predictions(model, examples, device, batch_size, shift)
            greedy = greedy_decode(model, examples, device, batch_size, shift)
            rows = [score_example(ex, tf, gr, split, step) for ex, tf, gr in zip(examples, tf_preds, greedy)]
            summary = summarize(rows, split, step)
            report.rows[split] = rows
            report.summary.extend(summary)
            report.metrics.update(wandb_metrics(summary))
            report.errors[split] = error_breakdown(rows, split, step)
            if out_dir is not None:
                out_dir = Path(out_dir)
                write_csv(out_dir / f"{split}.csv", rows, EXAMPLE_COLUMNS)
                conf_rows = [{"gold": g, "pred": p, "count": c} for (g, p), c in confusions(rows).most_common()]
                write_csv(out_dir / f"confusions_{split}.csv", conf_rows, ["gold", "pred", "count"])
                write_csv(out_dir / f"errors_{split}.csv", report.errors[split], ERROR_COLUMNS)
        if out_dir is not None:
            write_csv(Path(out_dir) / "summary.csv", report.summary, SUMMARY_COLUMNS)
    finally:
        if was_training:
            model.train()
    return report


# ----------------------------------------------------------------------------- CLI


def checkpoint_steps(run_dir):
    """Sorted [(step, checkpoint_dir)] under artifacts/<run>/checkpoints."""
    found = []
    for path in (run_dir / "checkpoints").glob("step_*"):
        if path.is_dir() and (path / "config.json").exists():
            found.append((int(path.name.split("_", 1)[1]), path))
    return sorted(found)


def select_steps(checkpoints, spec):
    if spec == "all":
        return checkpoints
    if spec == "last":
        return checkpoints[-1:]
    wanted = {int(s) for s in spec.split(",")}
    return [c for c in checkpoints if c[0] in wanted]


def rebuild_run_summary(run_dir):
    """Concatenate eval/step_*/summary.csv into eval/summary.csv and
    eval/step_*/errors_*.csv into eval/errors.csv."""
    rows = []
    for path in sorted((run_dir / "eval").glob("step_*/summary.csv")):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (int(r["step"]), r["split"], r["task"], r["attachment_site"], r["gap_type"]))
    write_csv(run_dir / "eval" / "summary.csv", rows, SUMMARY_COLUMNS)
    errors = []
    for path in sorted((run_dir / "eval").glob("step_*/errors_*.csv")):
        with open(path, newline="") as f:
            errors.extend(csv.DictReader(f))
    errors.sort(key=lambda r: (int(r["step"]), r["split"], r["task"], r["attachment_site"], r["gap_type"], ERROR_KINDS.index(r["kind"]), -int(r["count"])))
    write_csv(run_dir / "eval" / "errors.csv", errors, ERROR_COLUMNS)
    return rows


def errors_from_csv(run_dir, splits):
    """Rebuild errors_<split>.csv for every evaluated step from its <split>.csv
    (no model needed). Returns the last step's EvalReport-like errors dict."""
    last = None
    for step_dir in sorted((run_dir / "eval").glob("step_*")):
        step = int(step_dir.name.split("_", 1)[1])
        errors = {}
        for split in splits:
            path = step_dir / f"{split}.csv"
            if path.exists():
                errors[split] = error_breakdown(read_example_csv(path), split, step)
                write_csv(step_dir / f"errors_{split}.csv", errors[split], ERROR_COLUMNS)
        if errors:
            last = (step, errors)
    return last


def print_breakdown(report, top_confusions):
    print(f"  {'split':<10} {'task':<9} {'attachment':<10} {'gap':<12} {'n':>5} {'EM':>6} {'tok':>6} {'linear':>6} {'other':>6}")
    for r in report.summary:
        print(f"  {r['split']:<10} {r['task']:<9} {r['attachment_site']:<10} {r['gap_type']:<12} {r['n']:>5}"
              f" {r['exact_match']:>6.3f} {r['token_acc']:>6.3f} {r['linear_rate']:>6.3f} {r['other_rate']:>6.3f}")
    for split, rows in report.rows.items():
        top = confusions(rows).most_common(top_confusions)
        if top:
            print(f"  top mispredictions [{split}]: " + ", ".join(f"{g}->{p} x{c}" for (g, p), c in top))
    errors = [row for rows in report.errors.values() for row in rows]
    if errors:
        print("  error breakdown by construction:")
        print(format_error_breakdown(errors))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="*", help="glob over artifacts/ run names")
    parser.add_argument("--steps", default="all", help="'all', 'last', or comma-separated step numbers")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS))
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF repo id, or a local dir with {split}.csv files")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--shift", type=int, default=0, help="absolute-position shift (left pads) applied at eval")
    parser.add_argument("--skip-existing", action="store_true", help="skip steps whose eval/step_*/summary.csv exists")
    parser.add_argument("--top-confusions", type=int, default=10)
    parser.add_argument("--from-csv", action="store_true", help="only rebuild errors_*.csv from existing per-example CSVs (no model runs)")
    return parser.parse_args()


def main():
    args = parse_args()
    split_names = args.splits.split(",")
    run_dirs = [p for p in sorted(ARTIFACTS_DIR.glob(args.runs)) if (p / "checkpoints").is_dir()]
    if not run_dirs:
        raise SystemExit(f"no runs with checkpoints match artifacts/{args.runs}")

    if args.from_csv:
        for run_dir in run_dirs:
            print(f"== {run_dir.name}")
            last = errors_from_csv(run_dir, split_names)
            rebuild_run_summary(run_dir)
            if last is None:
                print("  no per-example CSVs found")
                continue
            step, errors = last
            print(f"  rebuilt errors_*.csv for every evaluated step; latest step {step:06d}:")
            print(format_error_breakdown([row for rows in errors.values() for row in rows]))
        return

    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    select_grammar(grammar_for(args.dataset))  # the vocabulary of the runs being evaluated
    splits = load_splits(args.dataset)
    examples = {name: prepare_examples(splits[name]) for name in split_names}

    from transformers import GPT2LMHeadModel

    for run_dir in run_dirs:
        print(f"== {run_dir.name}")
        last_report = None
        for step, ckpt in select_steps(checkpoint_steps(run_dir), args.steps):
            out_dir = run_dir / "eval" / f"step_{step:06d}"
            if args.skip_existing and (out_dir / "summary.csv").exists():
                continue
            try:
                model = GPT2LMHeadModel.from_pretrained(str(ckpt)).to(device)
            except Exception as err:  # noqa: BLE001 - a half-written checkpoint from a running job must not abort the sweep
                print(f"  step {step:06d}: could not load ({type(err).__name__}: {err}); skipping")
                continue
            last_report = run_eval(model, examples, device, out_dir, step, args.batch_size, args.shift)
            print(f"  step {step:06d} | {format_summary_line(last_report.summary)}", flush=True)
        rows = rebuild_run_summary(run_dir)
        print(f"  wrote {run_dir / 'eval' / 'summary.csv'} ({len(rows)} rows)")
        if last_report is not None:
            print_breakdown(last_report, args.top_confusions)


if __name__ == "__main__":
    main()
