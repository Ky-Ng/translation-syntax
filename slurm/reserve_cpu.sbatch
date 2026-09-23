#!/bin/bash
#SBATCH --job-name=dev-node
#SBATCH --partition=nlp_hiprio
#SBATCH --account=robinjia_875
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=7-00:00:00
#SBATCH --output=slurm_logs/dev-node-%j.out
#SBATCH --error=slurm_logs/dev-node-%j.err

# Reserve a CPU-only node for interactive work (launching jobs, running
# Claude Code, editing) for up to 7 days.
#
# NOTE: the nlp partitions typically cap walltime around 2 days; if this
# submission is rejected with a time-limit error, check the cap with
#   scontrol show partition nlp_hiprio | grep MaxTime
# and lower --time accordingly (or resubmit when it expires).
#
# Usage:
#   sbatch slurm/reserve_cpu.sbatch
#   squeue -u $USER                              # find the job id + node
#   srun --jobid=<JOBID> --overlap --pty bash    # drop a shell into the node
#
# From that shell you can `sbatch` GPU jobs, run `claude`, `uv run ...`, etc.

mkdir -p slurm_logs

module purge
module load gcc/13.3.0

cd $SLURM_SUBMIT_DIR

echo "dev node $(hostname) reserved at $(date); job $SLURM_JOB_ID"
sleep infinity
