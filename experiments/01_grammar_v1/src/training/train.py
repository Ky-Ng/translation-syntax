"""Train GPT2LMHeadModel on the ce-rb dataset (README "Model Architecture" spec).

Spec: 8 heads, {2,4,6} layers, d_model 128, word-level tokenizer from
src/cfg/grammar.py, batch 256, 300K steps, Adam lr 1e-4 with linear decay,
checkpoints every 2K steps to ./artifacts, random absolute-position shift
0-25 via padding, full-sequence LM loss, wandb run gpt2-8h<N>l300Ks4lr-<dataset>
(grouped by config, seed in the run name) in project translation-syntax, where
<dataset> is the HF repo name minus "grammar-v1-frames-" (ce-rb, ce, rb, ...),
"number-agreement-<variant>" for the grammar-number-agreement-frames-* repos,
or --dataset-tag. The vocabulary follows the dataset's grammar (--grammar
overrides training.data.grammar_for; recorded in launch_config.json).
At every checkpoint (or --eval-every) training.evaluate computes exact match,
token accuracy and the mispredicted tokens on validation + test, logs them to
wandb and writes the per-example CSVs to artifacts/<run>/eval/step_XXXXXX/.

Run (from experiments/01_grammar_v1/src):
    uv run python -m training.train --layers 2 --seed 42
"""

import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch

from training.data import (
    CONTEXT_LEN,
    DEFAULT_DATASET,
    GRAMMAR_LEXICONS,
    PAD_ID,
    STOI,
    VOCAB,
    collate,
    encode_split,
    grammar_for,
    load_splits,
    select_grammar,
)
from training.evaluate import format_summary_line, prepare_examples, run_eval

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layers", type=int, default=2, choices=(2, 4, 6))
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=300_000)
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=None, help="exact-match eval on validation+test every N steps (default: --checkpoint-every; 0 disables)")
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="HF repo id, or a local dir with {split}.csv files")
    parser.add_argument("--dataset-tag", default=None, help="dataset suffix for the run/group name (default: derived from --dataset)")
    parser.add_argument("--grammar", default=None, choices=tuple(GRAMMAR_LEXICONS), help="vocabulary grammar (default: derived from --dataset)")
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (for parallel CPU runs)")
    parser.add_argument("--wandb-mode", default="online", choices=("online", "offline", "disabled"))
    parser.add_argument("--wandb-project", default="translation-syntax")
    return parser.parse_args()


def dataset_tag(dataset, override=None):
    """Run-name suffix for a dataset: kylelovesllms/grammar-v1-frames-ce-rb -> ce-rb,
    kylelovesllms/grammar-number-agreement-frames-ce -> number-agreement-ce."""
    if override:
        return override
    name = Path(dataset).name
    if name.startswith("grammar-v1-frames-"):
        return name.removeprefix("grammar-v1-frames-")
    return name.removeprefix("grammar-").replace("-frames-", "-")


def evaluate(model, batches, device):
    model.eval()
    total_loss, total_batches = 0.0, 0
    with torch.no_grad():
        for batch in batches:
            out = model(**{k: v.to(device) for k, v in batch.items()})
            total_loss += out.loss.item()
            total_batches += 1
    model.train()
    return total_loss / max(total_batches, 1)


def main():
    args = parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    grammar = args.grammar or grammar_for(args.dataset)
    select_grammar(grammar)  # before any encoding and before the model's vocab_size is read

    steps_k = args.max_steps // 1000
    group = f"gpt2-{args.heads}h{args.layers}l{steps_k}Ks4lr-{dataset_tag(args.dataset, args.dataset_tag)}"  # README naming
    run_name = f"{group}-seed{args.seed}"
    run_dir = EXPERIMENT_DIR / "artifacts" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # per CLAUDE.md: save the launch config json in ./artifacts and in wandb
    config = {**vars(args), "grammar": grammar, "device": device, "run_name": run_name, "vocab_size": len(VOCAB)}
    config_path = run_dir / "launch_config.json"
    config_path.write_text(json.dumps(config, indent=2, default=str))

    import wandb

    wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=group,
        mode=args.wandb_mode,
        config={**config, "config_path": str(config_path)},
    )

    splits = load_splits(args.dataset)
    train_seqs = encode_split(splits["train"])
    val_seqs = encode_split(splits["validation"])
    val_batches = [
        collate(val_seqs[i : i + args.batch_size], random.Random(0))
        for i in range(0, len(val_seqs), args.batch_size)
    ]

    eval_every = args.checkpoint_every if args.eval_every is None else args.eval_every
    eval_examples = {name: prepare_examples(splits[name]) for name in ("validation", "test")} if eval_every else {}

    from transformers import GPT2Config, GPT2LMHeadModel

    model_config = GPT2Config(
        vocab_size=len(VOCAB),
        n_positions=CONTEXT_LEN,
        n_embd=args.d_model,
        n_layer=args.layers,
        n_head=args.heads,
        bos_token_id=STOI["<bos>"],
        eos_token_id=STOI["<eos>"],
        pad_token_id=PAD_ID,
    )
    model = GPT2LMHeadModel(model_config).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0 - step / args.max_steps)

    loss_csv = run_dir / "loss.csv"
    with open(loss_csv, "w", newline="") as f:
        csv.writer(f).writerow(["step", "split", "loss", "lr", "elapsed_s"])

    def log_row(step, split, loss):
        with open(loss_csv, "a", newline="") as f:
            csv.writer(f).writerow([step, split, f"{loss:.6f}", f"{scheduler.get_last_lr()[0]:.8f}", f"{time.time() - start:.1f}"])
        wandb.log({f"{split}/loss": loss, "lr": scheduler.get_last_lr()[0]}, step=step)

    order = list(range(len(train_seqs)))
    rng.shuffle(order)
    cursor = 0
    start = time.time()
    running_loss, running_count = 0.0, 0

    for step in range(1, args.max_steps + 1):
        if cursor + args.batch_size > len(order):
            rng.shuffle(order)
            cursor = 0
        batch_seqs = [train_seqs[i] for i in order[cursor : cursor + args.batch_size]]
        cursor += args.batch_size

        batch = collate(batch_seqs, rng)
        out = model(**{k: v.to(device) for k, v in batch.items()})
        out.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        running_loss += out.loss.item()
        running_count += 1
        if step % args.log_every == 0:
            log_row(step, "train", running_loss / running_count)
            running_loss, running_count = 0.0, 0

        if step % args.checkpoint_every == 0 or step == args.max_steps:
            log_row(step, "val", evaluate(model, val_batches, device))
            ckpt_dir = run_dir / "checkpoints" / f"step_{step:06d}"
            model.save_pretrained(ckpt_dir)
            steps_per_s = step / (time.time() - start)
            print(f"step {step}/{args.max_steps} ({steps_per_s:.1f} steps/s) -> {ckpt_dir}", flush=True)

        if eval_every and (step % eval_every == 0 or step == args.max_steps):
            report = run_eval(model, eval_examples, device, out_dir=run_dir / "eval" / f"step_{step:06d}", step=step, batch_size=args.eval_batch_size)
            wandb.log(report.metrics, step=step)
            print(f"step {step} eval | {format_summary_line(report.summary)}", flush=True)

    wandb.finish()
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
