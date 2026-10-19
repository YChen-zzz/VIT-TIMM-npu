# ViT-B/16 · CIFAR-10 Speedrun on timm · Ascend NPU

> This commit is the `record01_kerminal_migration` snapshot of the ViT-B/16 CIFAR-10 training speedrun built on [timm](https://github.com/huggingface/pytorch-image-models).
>
> The task trains `vit_base_patch16_224` from scratch on CIFAR-10 at 224×224 with **4 × 910C (8 dies)**, 200 epochs, seed 42 and a global batch of 512.
>
> The task was initially scoped as **model/operator-level optimization**, so the pre-declared quality metric is **final train loss** (`0.08390 ± 0.01`) together with **stable train time**. Eval top1 is reported for reference and is not the primary acceptance gate.
>
> - This commit contains the baseline training source, launcher and log.
> - Later records are kept as separate commits; the repository README on `main` lists the full record history.
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

The launcher does not download data, install dependencies or switch Python environments. Each run uses a unique experiment name and does not overwrite existing logs. The launcher requires all eight dies (four 910C NPUs) to be available.

## Record

Times are `stable_train_time_seconds` from `train_perf.json`: 200 epochs, excluding the first warmup epoch; validation and process startup are outside this region.

| Record | Time (stable train) | Final train loss | Eval top1 | Implementation | Log |
| - | - | - | - | - | - |
| `record01_kerminal_migration` | 1734.82 s | 0.084322795 | 84.01% | **[baseline]** original NPU port | [log](logs/record01_kerminal_migration_20261017_201347.log) |

The time above is the 2026-10-17 migration rerun. The historical validated time is 1743.26 s; migration changed it by -0.48%. The migration rerun's final train loss is 0.084322795 (historical loss +0.000414), the loss-curve mean relative error is 0.7572%, and eval top1 is 84.01% (historical 84.44%).

## Rules

New records must:

1. Reach final train loss within `0.08390 ± 0.01` on the same 200-epoch run, and keep the loss-curve mean relative error below 5%; eval top1 is recorded for reference, not used as the primary gate.
2. Keep the training task and dynamics unchanged: same model, data order, seed, batch size, epochs, learning-rate schedule and augmentation. Operator-level and precision changes are fair game.
3. Report the `stable_train_time_seconds` of one complete run from `train_perf.json` (first warmup epoch excluded), and include the full log (`train_perf.json`, `summary`, exit code), the training source and a `run.sh`.
4. Beat the prior record on the same hardware.

## Upstream

This project builds on [timm](https://github.com/huggingface/pytorch-image-models) and its `vit_base_patch16_224` model. The timm source and license are kept under `source/timm` and `source/TIMM_LICENSE`. All upstream credit belongs to the timm authors.
