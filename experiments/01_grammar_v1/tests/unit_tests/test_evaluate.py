"""Tests for training.evaluate: bookkeeping is checked against oracle models
whose behaviour is known exactly, plus a smoke test through a real GPT2."""

import csv
from types import SimpleNamespace

import torch
from training.data import PAD_ID, STOI, VOCAB, encode_row
from training.evaluate import (
    EXAMPLE_COLUMNS,
    format_summary_line,
    greedy_decode,
    linear_reversal,
    prepare_examples,
    run_eval,
    teacher_forced_predictions,
)

ROWS = [
    # ambiguous translate: linear == hierarchical
    {"Task": "translate", "HI Sentence": "cat chases dog", "HF Sentence": "cat dog chases",
     "Attachment Site": "None", "Gap Type": "None", "is_ambiguous": True, "HI Frame": "f0"},
    # unambiguous translate: hierarchical target differs from the linear reversal
    {"Task": "translate", "HI Sentence": "cat that bird fears chases dog", "HF Sentence": "bird fears that cat dog chases",
     "Attachment Site": "Subject", "Gap Type": "Object Gap", "is_ambiguous": False, "HI Frame": "f1"},
    {"Task": "copy", "HI Sentence": "dog swims", "HF Sentence": "dog swims",
     "Attachment Site": "None", "Gap Type": "None", "is_ambiguous": True, "HI Frame": "f2"},
]


class OracleModel(torch.nn.Module):
    """For each row's prefix ``<bos> src <task> <sep>`` emits ``target_of(row)``
    token by token (then <eos>, then <pad>) regardless of what actually follows
    <sep> in the input, so teacher-forced and greedy views can be predicted
    exactly. Assumes shift 0 / contiguous content."""

    def __init__(self, rows, target_of):
        super().__init__()
        self.table = {}
        for r in rows:
            prefix = tuple(encode_row(r["Task"], r["HI Sentence"], "")[:-1])  # drop the trailing <eos>
            self.table[prefix] = [STOI[w] for w in target_of(r).split()] + [STOI["<eos>"]]

    def forward(self, input_ids, attention_mask):
        batch, length = input_ids.shape
        logits = torch.zeros(batch, length, len(VOCAB))
        for b in range(batch):
            content = input_ids[b][attention_mask[b] == 1].tolist()
            sep = content.index(STOI["<sep>"])
            target = self.table[tuple(content[: sep + 1])]
            for pos in range(length):
                k = pos - sep  # position `sep` predicts the first target token
                nxt = target[k] if 0 <= k < len(target) else PAD_ID
                logits[b, pos, nxt] = 1.0
        return SimpleNamespace(logits=logits)


def test_linear_reversal_matches_readme_table():
    assert linear_reversal("cat chases dog") == "cat dog chases"
    assert linear_reversal("cat that bird fears chases dog") == "cat dog chases fears bird that"
    assert linear_reversal("cat chases dog that bird fears") == "cat fears bird that dog chases"
    assert linear_reversal("cat chases dog that likes bird") == "cat bird likes that dog chases"  # ambiguous
    assert linear_reversal("dog") == "dog"


def test_prepare_examples_splits_prefix_and_target():
    ex = prepare_examples(ROWS)[0]
    assert [VOCAB[i] for i in ex.prefix_ids] == ["<bos>", "cat", "chases", "dog", "<translate>", "<sep>"]
    assert [VOCAB[i] for i in ex.target_ids] == ["cat", "dog", "chases", "<eos>"]
    assert ex.meta["attachment_site"] == "None" and ex.meta["gap_type"] == "None"


def test_prepare_examples_normalizes_null_categories():
    # datasets' csv loader parses the literal "None" (no relative clause) as null
    row = {**ROWS[0], "Attachment Site": None, "Gap Type": None}
    ex = prepare_examples([row])[0]
    assert ex.meta["attachment_site"] == "None" and ex.meta["gap_type"] == "None"


def test_perfect_oracle_scores_everything_correct(tmp_path):
    model = OracleModel(ROWS, lambda r: r["HF Sentence"])
    report = run_eval(model, {"validation": prepare_examples(ROWS)}, "cpu", out_dir=tmp_path, step=7)
    assert report.metrics["validation/exact_match"] == 1.0
    assert report.metrics["validation/token_acc"] == 1.0
    assert report.metrics["validation/translate/linear_rate"] == 0.0
    for row in report.rows["validation"]:
        assert row["exact_match"] and row["n_wrong"] == 0 and row["mispredictions"] == "" and row["eos_emitted"]
        assert row["greedy"] == row["target"]
    assert [r["rule"] for r in report.rows["validation"]] == ["hierarchical", "hierarchical", "copy"]
    with open(tmp_path / "validation.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == EXAMPLE_COLUMNS
        assert len(list(reader)) == 3
    assert (tmp_path / "summary.csv").exists() and (tmp_path / "confusions_validation.csv").exists()


def test_linear_oracle_is_classified_as_linear_rule():
    def linear_target(r):
        return linear_reversal(r["HI Sentence"]) if r["Task"] == "translate" else r["HF Sentence"]

    model = OracleModel(ROWS, linear_target)
    report = run_eval(model, {"test": prepare_examples(ROWS)}, "cpu")
    ambiguous, unambiguous, copy = report.rows["test"]
    assert ambiguous["exact_match"] and ambiguous["rule"] == "hierarchical"
    assert copy["exact_match"] and copy["rule"] == "copy"
    assert not unambiguous["exact_match"] and unambiguous["rule"] == "linear"
    assert unambiguous["greedy"] == unambiguous["linear_target"] == "cat dog chases fears bird that"
    # teacher-forced mismatches are exactly the positions where the two targets differ
    gold, lin = ["bird", "fears", "that", "cat", "dog", "chases"], ["cat", "dog", "chases", "fears", "bird", "that"]
    expected = ";".join(f"{i}:{g}->{p}" for i, (g, p) in enumerate(zip(gold, lin), start=1) if g != p)
    assert unambiguous["mispredictions"] == expected
    assert unambiguous["n_wrong"] == 6 and unambiguous["n_target"] == 7
    assert report.metrics["test/translate/Subject_ObjectGap/linear_rate"] == 1.0
    assert report.metrics["test/translate/None_None/exact_match"] == 1.0


def test_swapped_target_reports_the_two_wrong_tokens():
    def swapped(r):
        words = r["HF Sentence"].split()
        if r["Task"] == "translate":
            words[0], words[1] = words[1], words[0]
        return " ".join(words)

    model = OracleModel(ROWS, swapped)
    row = run_eval(model, {"validation": prepare_examples(ROWS[:1])}, "cpu").rows["validation"][0]
    assert row["greedy"] == "dog cat chases" and row["eos_emitted"]
    assert row["mispredictions"] == "1:cat->dog;2:dog->cat"
    assert row["n_wrong"] == 2 and row["token_acc"] == 0.5 and row["rule"] == "other"


def test_summarize_groups_and_wandb_keys():
    model = OracleModel(ROWS, lambda r: r["HF Sentence"])
    report = run_eval(model, {"validation": prepare_examples(ROWS)}, "cpu", step=3)
    by_key = {(r["task"], r["attachment_site"], r["gap_type"]): r for r in report.summary}
    assert by_key[("all", "all", "all")]["n"] == 3
    assert by_key[("translate", "all", "all")]["n"] == 2
    assert by_key[("copy", "None", "None")]["n"] == 1
    assert all(r["step"] == 3 and r["split"] == "validation" for r in report.summary)
    assert {"validation/exact_match", "validation/copy/token_acc", "validation/translate/linear_rate"} <= set(report.metrics)
    assert format_summary_line(report.summary).startswith("validation: EM 1.000 tok 1.000 linear 0.000")


def test_real_gpt2_smoke(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    config = GPT2Config(vocab_size=len(VOCAB), n_positions=64, n_embd=16, n_layer=1, n_head=2,
                        bos_token_id=STOI["<bos>"], eos_token_id=STOI["<eos>"], pad_token_id=PAD_ID)
    model = GPT2LMHeadModel(config)
    model.train()
    examples = prepare_examples(ROWS)
    tf = teacher_forced_predictions(model, examples, "cpu", batch_size=2)
    greedy = greedy_decode(model, examples, "cpu", batch_size=2, max_new_tokens=5)
    assert [len(p) for p in tf] == [len(ex.target_ids) for ex in examples]
    assert all(1 <= len(g) <= 5 for g in greedy)
    report = run_eval(model, {"validation": examples, "test": examples}, "cpu", out_dir=tmp_path, step=1)
    assert model.training, "run_eval must restore train mode"
    for row in report.rows["test"]:
        assert row["n_wrong"] == round((1 - row["token_acc"]) * row["n_target"])
        assert row["exact_match"] == (row["n_wrong"] == 0)
    assert {p.name for p in tmp_path.iterdir()} == {"validation.csv", "test.csv", "confusions_validation.csv", "confusions_test.csv", "errors_validation.csv", "errors_test.csv", "summary.csv"}


def test_source_positions_and_error_type():
    from training.evaluate import error_type, source_positions

    assert source_positions("cat that bird fears chases dog", ["bird", "fears", "that", "cat", "dog", "chases"]) == "3 4 2 1 6 5"
    assert source_positions("cat chases dog", ["cat", "tickles", "cat"]) == "1 ? 1"
    base = {"exact_match": False, "rule": "other", "target": "laughs that newt sings", "eos_emitted": True}
    assert error_type({**base, "greedy": "newt sings that laughs"}) == "permutation"
    assert error_type({**base, "greedy": "newt sings that newt tickles"}) == "missing+repeat+foreign"
    assert error_type({**base, "greedy": "laughs that newt"}) == "missing"
    assert error_type({**base, "greedy": "laughs that newt sings sings", "eos_emitted": False}) == "repeat+no-eos"
    assert error_type({**base, "rule": "linear", "greedy": "x"}) == "linear"
    assert error_type({**base, "exact_match": True, "greedy": "laughs that newt sings"}) == "correct"


def test_error_breakdown_written_and_rebuildable(tmp_path):
    from training.evaluate import ERROR_COLUMNS, errors_from_csv, format_error_breakdown

    def linear_target(r):
        return linear_reversal(r["HI Sentence"]) if r["Task"] == "translate" else r["HF Sentence"]

    model = OracleModel(ROWS, linear_target)
    out_dir = tmp_path / "eval" / "step_000010"
    report = run_eval(model, {"test": prepare_examples(ROWS)}, "cpu", out_dir=out_dir, step=10)
    errors = report.errors["test"]
    by = lambda site, gap, kind: {r["key"]: r["count"] for r in errors if (r["attachment_site"], r["gap_type"], r["kind"]) == (site, gap, kind)}
    assert by("Subject", "Object Gap", "rule") == {"linear": 1}
    assert by("Subject", "Object Gap", "error_type") == {"linear": 1}
    assert by("Subject", "Object Gap", "first_error_position") == {"1": 1}
    assert by("Subject", "Object Gap", "ordering") == {"gold 3 4 2 1 6 5 -> model 1 6 5 4 3 2": 1}
    assert by("None", "None", "rule") == {"hierarchical": 1, "copy": 1}  # translate + copy rows, both correct
    assert by("None", "None", "error_type") == {}
    with open(out_dir / "errors_test.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ERROR_COLUMNS and len(list(reader)) == len(errors)
    # rebuilding from the per-example CSV reproduces the same breakdown
    (out_dir / "errors_test.csv").unlink()
    step, rebuilt = errors_from_csv(tmp_path, ["test"])
    assert step == 10 and (out_dir / "errors_test.csv").exists()
    assert [(r["kind"], r["key"], r["count"]) for r in rebuilt["test"]] == [(r["kind"], r["key"], r["count"]) for r in errors]
    text = format_error_breakdown(errors)
    assert "Subject / Object Gap" in text and "linear 1.00" in text
