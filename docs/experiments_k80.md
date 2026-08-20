# K80 training data and reproduced metrics

## Training inputs used by the release scripts

| File | Rows |
| --- | ---: |
| aug_ans_train.jsonl | 56992 |
| aug_ans_val.jsonl | 3010 |
| aug_qa_train.jsonl | 56995 |
| aug_qa_val.jsonl | 3007 |
| aug_rerank_train.jsonl | 57003 |
| aug_rerank_val.jsonl | 2998 |
| aug_rerank_hard_train_12000_union.jsonl | 12000 |
| aug_rerank_hard_train_12000_offloc.jsonl | 12000 |

Answer/QA grounding localizers are trained with
`ANS_GT_MAX_SAMPLES=20000` and `ANS_GT_MAX_VAL=500`.

## Reranker training

`scripts/train_reranker.sh` trains two contrastive path-answer rerankers on
`aug_rerank_hard_train_12000_union.jsonl` and
`aug_rerank_hard_train_12000_offloc.jsonl` (seed 0, 3000 steps, batch 1,
NEG=4, lr 1e-5, val-best selection), then weight-averages them 0.6/0.4.
Measured offline test SR: 80.35 (oracle 83.16), matching the submitted result.

## Inference configuration

The canonical K80 settings are in `scripts/infer.sh`:

```text
RERANK_K=80
RERANK_ALPHA=5
RERANK_BATCH=2
RERANK_TEXTS=tail,last,qa
ANS_CKPT=gtl_ans_ckpt_aug/snapshot5000.pth
ANS_CKPT2=gtl_ans_ckpt_qa/final.pth
RERANK_CKPT=gtl_rerank_ckpt_avg_uo_6/best.pth
LOC_CAND_K=20
WTA_MODE=ct_0.6_cap_3
```

## Expected results

| Split | SR | Oracle SR | DTC | Score |
| --- | ---: | ---: | ---: | ---: |
| test | 80.35 | 83.16 | 2.61 | offline |
| val_unseen | 74.69 | 77.18 | 2.34 | 0.7098 |
| val_seen | 76.92 | 82.42 | 1.76 | 0.7458 |
