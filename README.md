# FAN DialNav K80 Reproduction

This repository reproduces the final **FAN** submission for the DialNav
Challenge:

- Leaderboard method: `RAINbow-DST-GTL-Rerank-v9-Final-K80`
- Final submission file: `assets/fan/final/FAN_k80.json`

The method is a compliant natural-language dialog pipeline built on the
released RAINbow checkpoints. It adds answer/QA grounding, a larger candidate
pool, and a hard-negative path-answer reranker. It does **not** inject the
ground-truth path into the Navigator. After receiving the Guide's
natural-language answer, the Navigator uses its own answer/QA grounding and
path-answer reranking modules to plan the next waypoint along the scene graph;
the Guide communicates only natural language.

## Final results

| Split | SR | Oracle SR | DTC | Per-sample score |
| --- | ---: | ---: | ---: | ---: |
| test | 80.35 | 83.16 | 2.61 | offline |
| val_unseen | 74.69 | 77.18 | 2.34 | 0.7098 |
| val_seen | 76.92 | 82.42 | 1.76 | 0.7458 |

`test` is scored offline by the organizers and is the ranking metric; the
table above matches the public leaderboard entry. The end-to-end training
pipeline in this repository (`scripts/train_reranker.sh`) reaches the same
test SR (80.35). The repository includes local reproduction scripts for all
splits.

## Quickstart

```bash
cp .env.example .env
# Edit .env:
#   DIALNAV_PYTHON=/path/to/python
#   DIALNAV_MATTERPORT_SIM_BUILD=/path/to/Matterport3DSimulator/build

HF_REPO_ID=Njoker/FAN_DialNav_2026 bash scripts/download_hf_assets.sh   # fetch assets/ (weights, features, connectivity) from Hugging Face
bash scripts/setup.sh
bash scripts/verify.sh
bash scripts/run_all.sh
bash scripts/score.sh outputs/FAN_k80_repro.json
```

This repository is the code release; the large assets (official checkpoints,
features, connectivity, fine-tuned weights, and training data) are hosted on
Hugging Face under `Njoker/FAN_DialNav_2026` and downloaded into `assets/` by
the command above.

`scripts/run_all.sh` runs the K80 shards and merges them into
`outputs/FAN_k80_repro.json`. The expected output is the table above.

## Environment

- Python 3.10
- PyTorch 2.6.0 with CUDA 12.4
- Matterport3D Simulator built for Python 3.10

For a CUDA 12.4 conda environment:

```bash
conda env create -f environment.yml
conda activate fan-dialnav-k80
```

The Matterport3D Simulator must be built separately. `DIALNAV_MATTERPORT_SIM_BUILD`
should point to the simulator's `build/` directory, matching the
`PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH` instruction from RAINbow.

## Repository layout

```text
src/holistic/       K80 inference pipeline
src/modules/        DST, GTL, and LANA modules used by the pipeline
src/tools/          training, sharding, merging, and local scoring
scripts/            single entrypoint scripts
assets/base/        official RAINbow checkpoints and evaluation data
assets/fan/         K80 weights, exact training inputs, and final submission
outputs/            generated inference and merged submission files
```

No local paths are hardcoded. All paths are derived from `scripts/common.sh` and
the two required variables in `.env`.

## Training the custom K80 components

Answer, QA, and base-answer grounding training inputs are included under
`assets/fan/training_data/`.

### Reproducing the final test SR 80.35

There are two ways to obtain the final test result (SR 80.35):

1. **Reference (inference only, no training).** Run inference with the
   submitted leaderboard weights
   `assets/fan/weights/gtl_rerank_ckpt_hard/best_original.pth` (set
   `RERANK_CKPT_OVERRIDE` to that path). This reproduces the leaderboard
   numbers exactly (test 80.35 / val_unseen 74.69 / val_seen 76.92).
2. **End-to-end training.** `scripts/train_reranker.sh` trains two contrastive
   path-answer rerankers on the included 12k hard-negative candidate sets
   (`aug_rerank_hard_train_12000_union.jsonl` and
   `aug_rerank_hard_train_12000_offloc.jsonl`), weight-averages them 0.6/0.4,
   and produces `gtl_rerank_ckpt_final/best.pth`. Measured test SR of that
   pipeline: **80.35** (oracle 83.16), identical to the leaderboard test. To
   use the trained checkpoint, set `RERANK_CKPT_OVERRIDE` to its path and run
   `scripts/run_all.sh`; the provided checkpoint
   `gtl_rerank_ckpt_avg_uo_6/best.pth` is the default used by `scripts/infer.sh`.

```bash
bash scripts/train_ans.sh ans
bash scripts/train_ans.sh qa
bash scripts/train_reranker.sh     # end-to-end reranker training (test 80.35)
```

### Training data

All training inputs are included under `assets/fan/training_data/`: answer and
QA grounding sets, the 12k hard-negative candidate sets
(`aug_rerank_hard_train_12000_union.jsonl` and
`aug_rerank_hard_train_12000_offloc.jsonl`) used by `scripts/train_reranker.sh`,
and the reranker validation set `aug_rerank_val.jsonl`.

The expected training data row counts are listed in
`docs/experiments_k80.md`.

The optional raw-data builders are in `src/tools/optional_data_build/`. They are
provided for audit only. The included training files are the inputs used by the
release scripts.

## Compliance

- Only the official RAIN training split is used for custom training.
- `val_seen`, `val_unseen`, and `test` are used only for evaluation.
- Model weights are not updated during evaluation; test annotations are used
  only for final evaluation, while validation is used only for development
  evaluation.
- Navigator/Guide communication remains natural language.
- No ground-truth path is injected into the Navigator.
- This release contains only the final compliant method.

## Troubleshooting

- **Missing Matterport3D build**: set `DIALNAV_MATTERPORT_SIM_BUILD` in `.env`.
- **CUDA OOM**: reduce the number of parallel shards in `scripts/run_all.sh`.
- **HF offline errors**: the release uses offline tokenizers; do not remove
  `src/modules/qa/LANA/tokenizer_files/`.
- **Scene cache**: `scripts/common.sh` creates `./.cache/scene_cache`.
