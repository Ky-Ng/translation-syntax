"""Plot training/validation loss curves from artifacts/*/loss.csv.

Color encodes the layer config (fixed assignment, validated categorical
palette); linestyle encodes the split (solid train, dashed val); seeds within
a config share its hue. The loss.csv files remain the table view of the data.

Run (from experiments/01_grammar_v1/src):
    uv run python -m training.plot_loss                 # all runs in ./artifacts
    uv run python -m training.plot_loss --runs "gpt2-8h2l*"
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parents[2]

# categorical slots 1-3 (validated, all-pairs safe) in fixed assignment
LAYER_COLORS = {2: "#2a78d6", 4: "#eb6834", 6: "#1baf7a"}
FALLBACK_COLOR = "#eda100"
INK, INK_MUTED = "#333333", "#666666"


def read_run(loss_csv):
    by_split = defaultdict(lambda: ([], []))
    with open(loss_csv) as f:
        for row in csv.DictReader(f):
            steps, losses = by_split[row["split"]]
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
    return by_split


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="gpt2-*", help="glob of run dirs under ./artifacts")
    parser.add_argument("--out", type=Path, default=EXPERIMENT_DIR / "figures" / "training_loss.png")
    parser.add_argument("--log-x", action="store_true", help="log-scale steps (late-transition view)")
    parser.add_argument("--linear-y", action="store_true", help="linear loss axis instead of log")
    args = parser.parse_args()

    run_dirs = sorted(d for d in (EXPERIMENT_DIR / "artifacts").glob(args.runs) if (d / "loss.csv").exists())
    if not run_dirs:
        raise SystemExit(f"no runs with loss.csv match artifacts/{args.runs}")

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    seen_labels = set()
    for run_dir in run_dirs:
        layers_match = re.search(r"(\d+)l", run_dir.name)
        layers = int(layers_match.group(1)) if layers_match else None
        color = LAYER_COLORS.get(layers, FALLBACK_COLOR)
        for split, style in (("train", "-"), ("val", "--")):
            steps, losses = read_run(run_dir / "loss.csv")[split]
            if not steps:
                continue
            label = f"{layers}L {split}" if layers else f"{run_dir.name} {split}"
            ax.plot(
                steps,
                losses,
                style,
                color=color,
                linewidth=2,
                alpha=0.35 if label in seen_labels else 1.0,  # extra seeds recede
                label="_" + label if label in seen_labels else label,
            )
            seen_labels.add(label)

    if args.log_x:
        ax.set_xscale("log")
    if not args.linear_y:
        ax.set_yscale("log")
    ax.set_xlabel("Training step", color=INK)
    ax.set_ylabel("Cross-entropy loss" + ("" if args.linear_y else " (log)"), color=INK)
    ax.set_title(f"ce-rb training loss ({len(run_dirs)} run{'s' if len(run_dirs) > 1 else ''})", color=INK)
    ax.grid(True, color="#e6e6e6", linewidth=0.8)
    ax.tick_params(colors=INK_MUTED)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
