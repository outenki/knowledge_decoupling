# Project map

## Research focus

The current main comparison is between SML, RND_ID, and MASK. The project trains language models on the corresponding tokenized corpora and evaluates language behavior and knowledge-oriented tasks. Keep these conditions and their existing data/model paths distinct when reorganizing the code.

## Main entry points

| Purpose | Entry point | Notes |
| --- | --- | --- |
| Main causal-LM training | `src/train/train.py` | Hydra config selected with `--config-name`; implementation delegates to `src/lib/trainer.py`. |
| Dataset tokenization and block slicing | `src/data_processing/tokenize_and_slice_data.py` | Selects a text column, tokenizes it, then optionally slices it into fixed-size blocks. |
| Evaluation launchers | `scripts/eval/` | Shared evaluation commands and lm-eval task launchers. |
| hkd1 PBS jobs | `scripts/_jobs/train/hkd/`, `scripts/_jobs/eval/hkd/` | Submit with `qsub`; scripts use `#PBS` directives. |
| moon / Genkai PJM jobs | `scripts/_jobs/train/genkai/`, `scripts/_jobs/eval/genkai/` | Submit with `pjsub`; scripts use `#PJM` directives. |
| Shared Llama 3.2 1B training runtime | `scripts/_jobs/train/common/run_llama3_2_1b.sh` | PBS/PJM wrappers pass the Hydra config name and seed to this runner. Scheduler resources and log destinations remain in each wrapper. |

## Main comparison configs

| Condition | Current config file | Data name in config | Notes |
| --- | --- | --- | --- |
| SML | `config/train/Llama3.2_1B-sml.yaml` | `SmolLM2-135M-20B-sml-bs4096` | Seed is supplied by the job command in the existing Llama job scripts. |
| MASK | `config/train/Llama3.2_1B-sml-mask.yaml` | `SmolLM2-135M-20B-sml_mask-bs4096` | Separate tokenizer path is resolved under this dataset directory. |
| RND_ID | `config/train/Llama3.2_1B-sml_rnd_id.yaml` | `SmolLM2-135M-20B-core_rnd_id-bs4096` | This is the RND_ID condition generated from the SmolLM2 corpus. The legacy corpus directory name is retained for path compatibility; `core_rnd_id` config name remains as an alias. |

The three configs inherit common Llama 3.2 1B model and training settings from `config/train/_base/llama3_2_1b.yaml`; each condition supplies its own data name and W&B metadata. RND_ID's dataset lineage is SmolLM2-based, so `sml_rnd_id` is the canonical experiment label. The existing corpus directory string remains unchanged to avoid breaking data loading.

## Current scheduler split

- hkd1 is PBS: `qsub`, `qstat`, `qdel`.
- moon / Genkai is PJM: `pjsub`, `pjstat`, `pjdel`.

Scheduler directives are not interchangeable. Keep resource declarations in scheduler-specific job files while keeping experiment parameters in the Hydra configs. The current Llama 3.2 1B wrappers request 4 GPUs on both clusters, with 160 hours on hkd1 and 100 hours on moon; the shared runner handles module setup, environment loading, working directory, and the Python command.

## Suggested maintenance order

1. Identify the active job scripts for all three conditions and map legacy RND_ID paths to the SmolLM2-based corpus.
2. Update remaining job scripts to call canonical config names while retaining compatibility aliases.
3. Consolidate PBS and PJM launch scripts only after mapping their resource and environment differences.
4. Split the training entry point by responsibility without changing training behavior in the same change.

