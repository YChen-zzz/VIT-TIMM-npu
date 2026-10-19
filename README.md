# ViT-B/16 · CIFAR-10 Speedrun on timm · Ascend NPU Leaderboard

> **This repository tracks NPU reproductions of a ViT-B/16 CIFAR-10 training speedrun built on [timm](https://github.com/huggingface/pytorch-image-models).**
>
> The task trains `vit_base_patch16_224` from scratch on CIFAR-10 at 224×224 with **4 × 910C (8 dies)**, 200 epochs, seed 42 and a global batch of 512. Two implementations are recorded as separate commits: the pre-optimization `record01_kerminal_migration` and the fastest fully validated `record02_optimization` build selected from five complete optimization runs.
>
> The task was initially scoped as **model/operator-level optimization**, so the pre-declared quality metric is **final train loss** (`0.08390 ± 0.01`) together with **stable train time**. Eval top1 is reported for reference and is not the primary acceptance gate.
>
> - The leaderboard record history below is the main result of this repository.
> - The current tree contains the latest record, `record02_optimization`; `record01_kerminal_migration` is available at its git tag.
> - Upstream timm code, license and attribution are preserved under `source/`.

---

## Runtime

The NPU runs use a common reference runtime with four Ascend 910C NPUs (8 dies):

| Field | Value |
|---|---|
| Python | 3.10.19 |
| torch | `2.7.1+cpu` |
| torch_npu | `2.7.1.post2` |
| torchvision | `0.22.1` |
| CANN | `8.5.0` |
| Hardware | 4 × Ascend 910C (8 dies) |
| Process count | 8 ranks in one `torchrun` job |
| Model | `vit_base_patch16_224`, 10 classes, from scratch |
| Data | CIFAR-10, 224×224 |
| Batch | 64 per rank, global 512 |
| Training | 200 epochs, seed 42, BF16 AMP |
| Optimizer | Adam, lr 1e-4, eps 1e-8, wd 0, cosine, warmup 0 |
| Augmentation | `rand-m14-n2`, horizontal flip 0.5 |
| Default image | `docker.cnb.cool/nilpotenter/docker/codeserver-mindspeed:v1.0.5` |

`torchrun --nproc_per_node=8` launches eight ranks (one per die) for one training job; it does not mean eight independent trainings. The `+cpu` suffix on the torch version is the Ascend package build string and does not mean the training ran on CPU.

## Running

The training data is not included in this repository. Set `DATA_DIR` to a CIFAR-10 directory containing `cifar-10-batches-py/data_batch_1`:

```bash
DATA_DIR=/path/to/cifar10 bash run.sh
```

To reproduce the baseline record, check out its tag first:

```bash
git checkout record01_kerminal_migration
DATA_DIR=/path/to/cifar10 bash run.sh
git checkout main
```

The launchers do not download data, install dependencies or switch Python environments. Each run uses a unique experiment name and does not overwrite existing logs. The launchers require all eight dies (four 910C NPUs) to be available.

## NPU leaderboard record history

Times are `stable_train_time_seconds` from `train_perf.json`: 200 epochs, excluding the first warmup epoch; validation and process startup are outside this region. Rows are sorted from longest to shortest. Each row is one complete 200-epoch run, not an average over seeds.

| # | Record | Time (stable train) | Final train loss | Eval top1 | Implementation | Log |
| - | - | - | - | - | - | - |
| 1 | [record01_kerminal_migration](../../tree/record01_kerminal_migration) | 1734.82 s | 0.084322795 | 84.01% | **[baseline]** original NPU port | [log](logs/record01_kerminal_migration_20261017_201347.log) |
| 2 | [record02_optimization](../../tree/record02_optimization) | **1386.06 s** | 0.083367832 | 84.22% | **[optimized]** patchify Linear, async gradient buckets, flat-gradient view, deferred loss metrics, bucket-local Adam | [log](logs/record02_optimization_20261017_201347.log) |

The two records are tagged `record01_kerminal_migration` and `record02_optimization`. The times above are the 2026-10-17 migration reruns. Historical validated times are 1743.26 s (record01_kerminal_migration) and 1392.25 s (record02_optimization); migration changed them by -0.48% and -0.44%. record02_optimization is 20.10% faster than the migrated record01_kerminal_migration. For context, the complete process wall clock including validation and startup was 1889.6 s and 1539.4 s.

Both records exit with code 0 and pass the pre-declared gate:

- record01_kerminal_migration: final train loss 0.084322795 (historical loss +0.000414); loss-curve mean relative error 0.7572%; eval top1 84.44% → 84.01%.
- record02_optimization: final train loss 0.083367832, identical to the historical loss; loss-curve maximum absolute error 5.96e-8; eval top1 84.22% unchanged.

## Rules

New records must:

1. Reach final train loss within `0.08390 ± 0.01` on the same 200-epoch run, and keep the loss-curve mean relative error below 5%; eval top1 is recorded for reference, not used as the primary gate.
2. Keep the training task and dynamics unchanged: same model, data order, seed, batch size, epochs, learning-rate schedule and augmentation. Operator-level and precision changes are fair game.
3. Report the `stable_train_time_seconds` of one complete run from `train_perf.json` (first warmup epoch excluded), and include the full log (`train_perf.json`, `summary`, exit code), the training source and a `run.sh`.
4. Beat the prior record on the same hardware.

Other than that, anything goes: NPU-specific optimisation is the point of this repository.

## Submitting

Open a pull request against this repository. Each submission adds its training source, a short `run.sh`, the full log, hardware and software versions, and its new leaderboard row. Do not include CIFAR-10 data, model weights, passwords, private keys or tokens.

## Upstream

This project builds on [timm](https://github.com/huggingface/pytorch-image-models) and its `vit_base_patch16_224` model. The timm source and license used by each record are kept under `source/timm` and `source/TIMM_LICENSE`. All upstream credit belongs to the timm authors. This repository's records are NPU training results for this task, not an official timm benchmark.
