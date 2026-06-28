# Reproduce From Zero

This note records the exact assets used for the completed local rerun.

## 1. Data

Expected local data root:

```text
/home/xjtluaiac2024/Desktop/ECCV/
  Synapse/*.h5
  AMOS/*_image.npy
  AMOS/*_label.npy
  amos_splits/*.txt
```

Synapse CPS uses the split files committed under `splits/synapse/*_used.txt`.
AMOS MagicNet uses the split files committed under `splits/amos/`.

Important: `ECCV/synapse_splits/test.txt` is not the split used for the
Synapse CPS checkpoint in this package. Use `splits/synapse/test_used.txt`.

## 2. Source Code

Synapse CPS evaluation uses the CPS/VNet utilities from:

```text
/home/xjtluaiac2024/Desktop/GAcps复现
```

AMOS MagicNet evaluation uses the MagicNet code from:

```text
/home/xjtluaiac2024/Desktop/MagicNet复现
```

The wrapper script in this repository is:

```bash
python scripts/run_current_best_eval.py --help
```

## 3. Checkpoints

See `docs/WEIGHTS_MANIFEST.md`.

Recommended local placement:

```text
weights/Synapse_CPS_baseline_syn20_local/
  iter_11500_dice_0.632039_best_A.pth
  iter_11500_dice_0.632039_best_B.pth

weights/AMOS_MagicNet_GA_10labeled/
  iter_15000_dice_0.544700_best.pth
```

## 4. Commands Used For This Rerun

Synapse CPS full test rerun on CPU:

```bash
python scripts/run_current_best_eval.py \
  --exp synapse_cps \
  --run-inference \
  --device cpu \
  --out-dir eval_outputs_synapse_cpu_full
```

AMOS MagicNet full test rerun on CPU:

```bash
python /home/xjtluaiac2024/Desktop/MagicNet复现/code/eval_amos_magicnet.py \
  --root_path /home/xjtluaiac2024/Desktop/ECCV/AMOS \
  --split /home/xjtluaiac2024/Desktop/ECCV/amos_splits/test.txt \
  --checkpoint /home/xjtluaiac2024/Desktop/MagicNet复现/results/magicnet_baseline_all_datasets_bundle_2026-06-28/results/AMOS_MagicNet_GA_10labeled/iter_15000_dice_0.544700_best.pth \
  --out_dir eval_outputs_amos_cpu_full \
  --device cpu
```

Merge final tables:

```bash
python scripts/run_current_best_eval.py --exp all --out-dir eval_outputs
```

The final manually merged rerun tables are committed in `results/`.

## 5. Final Rerun Summary

| Experiment | Cases | Organs | DSC | ASD | HD95 | NSD |
|---|---:|---:|---:|---:|---:|---:|
| Synapse_CPS_baseline_syn20_local | 6 | 13 | 0.6628 | 5.4448 | 12.3007 | 0.7656 |
| AMOS_MagicNet_GA_10labeled | 120 | 15 | 0.6347 | 5.8763 | 11.2586 | 0.7284 |

Detailed organ-level metrics:

- `results/final_rerun_detailed_dsc_asd_hd95_nsd.md`
- `results/final_rerun_detailed_dsc_asd_hd95_nsd.csv`

Raw per-case metric arrays:

- `results/synapse_cps_rerun_metric_final.npy`
- `results/amos_magicnet_rerun_metric_final.npy`

## 6. Runtime Notes

This machine had no usable NVIDIA driver during the rerun, so evaluation was
performed on CPU.

Observed runtime:

| Experiment | Device | Runtime |
|---|---|---:|
| Synapse CPS | CPU | about 22 minutes |
| AMOS MagicNet | CPU | about 2 hours 15 minutes |

On a CUDA machine the same commands can be run with `--device cuda`.
