# FAN DialNav K80 Reproduction

This repository reproduces the final **FAN** submission for the DialNav
Challenge:

- Leaderboard method: `RAINbow-DST-GTL-Rerank-v9-Final-K80`
- Final submission file: `assets/fan/final/FAN_k80.json`

The method is a compliant natural-language dialog pipeline built on the
released RAINbow checkpoints. It adds answer/QA grounding, a larger candidate
pool, and a path-answer reranker (a 0.6/0.4 weight average of two contrastive
rerankers trained on hard-negative candidate paths). It does **not** inject
the ground-truth path into the Navigator. After receiving the Guide's
natural-language answer, the Navigator uses its own answer/QA grounding and
path-answer reranking modules to plan the next waypoint along the scene graph;
the Guide communicates only natural language. The Guide knows the destination
by task design and conveys it to the Navigator only through natural language.

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

HF_REPO_ID=Njoker/FAN_DialNav_2026 bash scripts/download_hf_assets.sh   # for code-only GitHub checkouts: fetch assets/ from Hugging Face
bash scripts/setup.sh
bash scripts/verify.sh
bash scripts/run_all.sh
bash scripts/score.sh outputs/FAN_k80_repro.json
```

`scripts/run_all.sh` runs the K80 shards and merges them into
`outputs/FAN_k80_repro.json`. The offline test SR (80.35), the ranking metric,
is reproduced exactly; development-split SR may vary by about +/-1 across
checkpoints. The expected output is the table above.

## Resources and runtime

- Inference (`scripts/run_all.sh`) runs 16 shards in parallel and defaults to
  **8 GPUs**; set `GPU_COUNT` to use fewer (shards are launched in waves).
  Expected wall time is roughly 1 hour on 8 GPUs.
- Training (`scripts/train_reranker.sh`) needs **1 GPU** and takes roughly
  one hour (two 3,000-step reranker trainings plus the weight average).
- The full inference configuration (candidate pool K, text variants,
  localization candidates, WTA policy, etc.) is listed in
  `docs/experiments_k80.md`.

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
FAN_k80_release
|-- README.md               this document
|-- .env.example            environment template (python + simulator paths)
|-- environment.yml         conda environment
|-- requirements.txt        python dependencies
|-- LICENSE / NOTICE / DATA_LICENSE.md
|-- scripts/
|   |-- common.sh           path/env setup shared by all scripts
|   |-- download_hf_assets.sh  fetch assets/ from Hugging Face (code-only checkouts)
|   |-- setup.sh / verify.sh   environment checks and asset verification
|   |-- run_all.sh          full K80 inference (per-shard scene caches)
|   |-- make_shards.sh / merge.sh / score.sh   sharding, merging, local scoring
|   |-- train_ans.sh        answer/QA localizer training (ans | qa)
|   `-- train_reranker.sh   end-to-end reranker training (union + offloc, 0.6/0.4 average)
|-- src/
|   |-- holistic/           K80 inference pipeline (main.py, ModularGuide/Navigator, evaluator, holistic_models/)
|   |-- modules/            DST, GTL, and LANA modules used by the pipeline
|   `-- tools/              training, sharding, merging, and local scoring scripts
|-- assets/                 populated by download_hf_assets.sh (or present in the self-contained package)
|   |-- base/dataset/       official RAINbow checkpoints, features, connectivity, annotations
|   |-- fan/weights/        fine-tuned localizers and the final reranker
|   |-- fan/training_data/  answer/QA data and 12k hard-negative candidate sets
|   |-- fan/final/          FAN_k80.json (submission file)
|   `-- manifest/           assets.sha256 and release_manifest.json
|-- hf_manifest/            checksums and manifest for the Hugging Face assets
`-- outputs/                generated inference and merged submission files (gitignored)
```

`assets/` is excluded from this code repository and is fetched from Hugging
Face (`Njoker/FAN_DialNav_2026`) by `scripts/download_hf_assets.sh`; the
self-contained package includes it directly.

The FAN pipeline derives all runtime paths from `scripts/common.sh` and the two
required variables in `.env`; upstream RAINbow module defaults are overridden
by the pipeline's command-line arguments.

## Training the custom K80 components

Answer and QA grounding training inputs are included under
`assets/fan/training_data/`. All runs use seed 0.

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

All training inputs are included in the Hugging Face assets (downloaded by
`scripts/download_hf_assets.sh`) and, in the self-contained package, under
`assets/fan/training_data/`: answer and QA grounding sets, the 12k
hard-negative candidate sets (`aug_rerank_hard_train_12000_union.jsonl` and
`aug_rerank_hard_train_12000_offloc.jsonl`) used by
`scripts/train_reranker.sh`, and the reranker validation set
`aug_rerank_val.jsonl`.

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
