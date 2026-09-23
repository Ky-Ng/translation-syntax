#!/bin/bash
#SBATCH --job-name=ce-rb-train
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/ce-rb-%A_%a.out
#SBATCH --error=slurm_logs/ce-rb-%A_%a.err

# README sweep: 3 layer counts x 30 seeds (42-71) = 90 runs.
#
# Usage:
#   sbatch --array=0-89 slurm/train_ce_rb.sbatch          # full sweep
#   sbatch --array=0 slurm/train_ce_rb.sbatch             # single run: 2 layers, seed 42
#   sbatch --array=0-89 --partition=nlp slurm/train_ce_rb.sbatch   # preemptable
#
# Array task t maps to: layers = (2,4,6)[t / 30], seed = 42 + t % 30.
# Extra flags are forwarded to training.train (e.g. --max-steps 1000 to smoke test).

mkdir -p slurm_logs

module purge
module load gcc/13.3.0
module load cuda/12.4.0

LAYERS_LIST=(2 4 6)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
LAYERS=${LAYERS_LIST[$((TASK_ID / 30))]}
SEED=$((42 + TASK_ID % 30))

cd "$SLURM_SUBMIT_DIR/experiments/01_grammar_v1/src"
uv run python -m training.train --layers "$LAYERS" --seed "$SEED" "$@"
