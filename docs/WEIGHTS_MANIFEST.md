# Best Checkpoint Manifest

This repository intentionally does not commit `.pth` tensors because the
checkpoints are large binary artifacts. Store them with Git LFS or release
assets if they need to be hosted on GitHub.

## Synapse CPS Baseline, 20% Labels

Experiment: `Synapse_CPS_baseline_syn20_local`

Best validation checkpoint pair:

| Model | File | Size | SHA256 | Local source |
|---|---:|---:|---|---|
| A | `iter_11500_dice_0.632039_best_A.pth` | 37,887,271 bytes | `a607c3f8a307186703b423d7940aa3ca927c3cd7dcd3d997bd5fbe2b8b5a31ac` | `每日日志/cps_local_bundle_2026-06-26/output/baseline/Synapse_CPS_baseline_syn20_local_GA_4labeled_seed_1337/iter_11500_dice_0.632039_best_A.pth` |
| B | `iter_11500_dice_0.632039_best_B.pth` | 37,887,271 bytes | `a7533500fc47c78f60086aad9a2986db75257c2bb7e4d3b067c6567284740221` | `每日日志/cps_local_bundle_2026-06-26/output/baseline/Synapse_CPS_baseline_syn20_local_GA_4labeled_seed_1337/iter_11500_dice_0.632039_best_B.pth` |

Expected placement for rerun:

```text
weights/Synapse_CPS_baseline_syn20_local/
  iter_11500_dice_0.632039_best_A.pth
  iter_11500_dice_0.632039_best_B.pth
```

## AMOS MagicNet Baseline, 10 Labels

Experiment: `AMOS_MagicNet_GA_10labeled`

Best validation checkpoint:

| Model | File | Size | SHA256 | Local source |
|---|---:|---:|---|---|
| MagicNet | `iter_15000_dice_0.544700_best.pth` | 72,405,523 bytes | `21d38922d7899c1ae74fccbdf9dc53ce707cc86cb857f7f4b73d65ef9605366d` | `/home/xjtluaiac2024/Desktop/MagicNet复现/results/magicnet_baseline_all_datasets_bundle_2026-06-28/results/AMOS_MagicNet_GA_10labeled/iter_15000_dice_0.544700_best.pth` |

Expected placement for rerun:

```text
weights/AMOS_MagicNet_GA_10labeled/
  iter_15000_dice_0.544700_best.pth
```

## Verification

```bash
sha256sum weights/Synapse_CPS_baseline_syn20_local/*.pth
sha256sum weights/AMOS_MagicNet_GA_10labeled/*.pth
```
