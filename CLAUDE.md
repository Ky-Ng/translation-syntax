# Allegro Project — Claude Instructions

## Environment

- USC CARC SLURM cluster
- Package manager: uv (NOT conda, NOT pip directly)
- Python: managed by uv, pinned in pyproject.toml (>=3.11)
- Virtual env: `.venv/` at project root (managed by uv)

## Cluster Info

- **Partitions:** `nlp_hiprio` (high priority, no preemption) and `nlp` (preemptable)
- **Account:** `robinjia_875`
- **GPUs:** ~60 A6000s, 4 A100s
- **Default resources per job:** 8 CPUs, 1 GPU, 40G RAM
- **Max walltime:** typically 2 days default in templates

## How to Run Things

```bash
uv sync                                                          # install dependencies
uv run python experiments/<experiment>/run.py                    # run locally
uv run pytest                                                    # run tests
sbatch slurm/run_gpu.sbatch experiments/00_example/run.py        # submit job (high priority)
sbatch slurm/run_preempt.sbatch experiments/00_example/run.py    # submit job (preemptable)
sbatch --array=0-2 slurm/run_array.sbatch                       # array job
```

Array jobs set `SLURM_ARRAY_TASK_ID` env var; experiments can read it via `os.environ.get("SLURM_ARRAY_TASK_ID")`.

If `which slurm` returns `slurm not found`, this means that you may be running locally on Mac. You may use CPU or MPS whichever which you find to be more efficient and run jobs in parallel. Please be mindful of system RAM and storage when on Mac.

## Project Layout

- `experiments/` — numbered experiment folders (00_xxx/, 01_xxx/, ...) each with `run.py`, `results/`, `logs/`, `figures/`
- `data/` — datasets (gitignored)
- `notes/` — writeups, paper summaries, meeting notes (see `notes/README.md`)
- `slurm/` — SLURM job templates

## Conventions

- Experiments: numbered folders in `experiments/` (00_xxx/, 01_xxx/, ...) each with a `run.py`. Create new folders, don't edit old ones.
- Each experiment folder should have a `README.md` with detailed setup, results, and observations. The top-level `experiments/README.md` should only contain brief descriptions of each experiment.
- Tracking: wandb (disable with `WANDB_MODE=disabled`)
- Do NOT read files in `notes/discussions/` (meeting notes) unless explicitly requested by the user

## Collaboration Notes

- Pushback if you feel that there is something that is not clear. You should be taking your high level instructions mainly from README.md and design docs which we collaborate on.

- Pushback for PRs which should be merged once an experiment is finished.

- When working on multi-step tasks, break the problem into steps and write thorough unit tests and integration tests. Re-run these tests when you modify source code to ensure we are on the same page. 

    - After writing and running tests, visualize whether or not the tests have passed in our chat. In the future we should make a Web UI to visualize the results of experiments along with tests.


- Use modular design and avoid rewriting code within an experiment folder; push back for a refactor when needed. Use infrastructure as code principles, e.g. use a python file for constants like the Context Free Grammar or .json files for model configs.

- When creating HuggingFace datasets, you can use ./data as staging before uploading to my account, "kylelovesllms". Use a dataset collection named `translation-syntax-datasets-<EXPERIMENT_NUMBER>` (e.g. `translation-syntax-datasets-01_grammar_v1`)

- When saving model checkpoints, save them in the experiments ./artifacts folder as a staging before uploading to my account, "kylelovesllms". Use a model collection named `translation-syntax-models-<EXPERIMENT_NUMBER>-<MODEL>-<STEPS>` (e.g. `translation-syntax-01_grammar_v1-gpt2-2l2h128d10ep256bsz3lr01drop-1000`) 

- Log training runs to Wandb. Use filters programatically and ask me often which Project I would like to put the wandb output run into. Specifically, I prefer having more fine grained Groups (e.g. model, seed). Different datasets and steps should be logged as separate Projects named `<Model>-<Dataset>-<Steps>`

    - Model run json launch scripts should be saved in ./artifacts and in the Wandb config (Wandb should also have the path to the config script)

    - Finished model runs should also have a `Report-<Model>-<Dataset>-<Steps>.md` written in the results folder.
