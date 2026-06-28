#!/usr/bin/env python3
"""Evaluate the current CPS/MagicNet evidence bundle.

By default this script rebuilds detailed metric tables from the saved
per-case metric arrays. With ``--run-inference`` it loads the best checkpoints
and runs the original GAcps validation utilities against the ECCV dataset.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parent
if ROOT.name == "scripts":
    ROOT = ROOT.parent
ECCV_ROOT = Path("/home/xjtluaiac2024/Desktop/ECCV")
GACPS_ROOT = Path("/home/xjtluaiac2024/Desktop/GAcps复现")
AMOS_BUNDLE_ROOT = (
    Path("/home/xjtluaiac2024/Desktop/MagicNet复现/results")
    / "magicnet_baseline_all_datasets_bundle_2026-06-28/results/AMOS_MagicNet_GA_10labeled"
)

SYNAPSE_TEST = ["0004", "0007", "0010", "0033", "0035", "0036"]
SYNAPSE_ORGANS = [
    "spleen",
    "r.kidney",
    "l.kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "ivc",
    "portal and splenic vein",
    "pancreas",
    "right adrenal gland",
    "Left adrenal gland",
]
AMOS_ORGANS = [
    "spleen",
    "r.kidney",
    "l.kidney",
    "gallbladder",
    "esophagus",
    "liver",
    "stomach",
    "aorta",
    "ivc",
    "pancreas",
    "right adrenal gland",
    "Left adrenal gland",
    "duodenum",
    "bladder",
    "prostate/uterus",
]
METRIC_NAMES = ["DSC", "HD95", "NSD", "ASD"]

EXPERIMENTS = {
    "synapse_cps": {
        "title": "Synapse_CPS_baseline_syn20_local",
        "type": "cps",
        "num_classes": 14,
        "base_dir": ECCV_ROOT / "Synapse",
        "test_list": SYNAPSE_TEST,
        "organs": SYNAPSE_ORGANS,
        "metric_npy": ROOT / "results/synapse_cps_rerun_metric_final.npy",
        "metric_npy_fallback": ROOT
        / "每日日志/cps_local_bundle_2026-06-26/output/baseline/"
        / "Synapse_CPS_baseline_syn20_local_GA_4labeled_seed_1337"
        / "metric_final_Synapse_CPS_baseline_syn20_local.npy",
        "ckpt_a": ROOT / "weights/Synapse_CPS_baseline_syn20_local/iter_11500_dice_0.632039_best_A.pth",
        "ckpt_a_fallback": ROOT
        / "每日日志/cps_local_bundle_2026-06-26/output/baseline/"
        / "Synapse_CPS_baseline_syn20_local_GA_4labeled_seed_1337"
        / "iter_11500_dice_0.632039_best_A.pth",
        "ckpt_b": ROOT / "weights/Synapse_CPS_baseline_syn20_local/iter_11500_dice_0.632039_best_B.pth",
        "ckpt_b_fallback": ROOT
        / "每日日志/cps_local_bundle_2026-06-26/output/baseline/"
        / "Synapse_CPS_baseline_syn20_local_GA_4labeled_seed_1337"
        / "iter_11500_dice_0.632039_best_B.pth",
    },
    "amos_magicnet": {
        "title": "AMOS_MagicNet_GA_10labeled",
        "type": "magicnet",
        "num_classes": 16,
        "base_dir": ECCV_ROOT / "AMOS",
        "test_list_file": ECCV_ROOT / "amos_splits/test.txt",
        "organs": AMOS_ORGANS,
        "metric_npy": ROOT / "results/amos_magicnet_rerun_metric_final.npy",
        "metric_npy_fallback": AMOS_BUNDLE_ROOT / "metric_final_AMOS_MagicNet.npy",
        "ckpt": ROOT / "weights/AMOS_MagicNet_GA_10labeled/iter_15000_dice_0.544700_best.pth",
        "ckpt_fallback": AMOS_BUNDLE_ROOT / "iter_15000_dice_0.544700_best.pth",
    },
}


def read_test_list(cfg: dict) -> list[str]:
    if "test_list" in cfg:
        return list(cfg["test_list"])
    with open(cfg["test_list_file"], encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def metric_rows(name: str, metric_final: np.ndarray) -> tuple[list[dict], dict]:
    cfg = EXPERIMENTS[name]
    mean = metric_final.mean(axis=0)
    std = metric_final.std(axis=0)
    rows = []
    for organ_idx, organ in enumerate(cfg["organs"]):
        row = {"Experiment": cfg["title"], "Organ": organ}
        for metric_idx, metric_name in enumerate(METRIC_NAMES):
            row[f"{metric_name}_mean"] = mean[metric_idx, organ_idx]
            row[f"{metric_name}_std"] = std[metric_idx, organ_idx]
            row[f"{metric_name}_mean±std"] = (
                f"{mean[metric_idx, organ_idx]:.4f}±{std[metric_idx, organ_idx]:.4f}"
            )
        rows.append(row)

    summary = {
        "Experiment": cfg["title"],
        "Cases": metric_final.shape[0],
        "Organs": metric_final.shape[2],
        "DSC_mean": mean[0].mean(),
        "HD95_mean": mean[1].mean(),
        "NSD_mean": mean[2].mean(),
        "ASD_mean": mean[3].mean(),
    }
    return rows, summary


def write_tables(metrics_by_exp: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []
    for name, metric_final in metrics_by_exp.items():
        rows, summary = metric_rows(name, metric_final)
        all_rows.extend(rows)
        summaries.append(summary)

    detail = pd.DataFrame(all_rows)
    summary = pd.DataFrame(summaries)
    for df in (detail, summary):
        for col in df.select_dtypes(include="number").columns:
            if col not in ("Cases", "Organs"):
                df[col] = df[col].round(4)

    detail.to_csv(out_dir / "detailed_dsc_hd95_nsd_asd_table.csv", index=False)
    summary.to_csv(out_dir / "summary_dsc_hd95_nsd_asd.csv", index=False)

    compact_cols = [
        "Experiment",
        "Organ",
        "DSC_mean±std",
        "HD95_mean±std",
        "NSD_mean±std",
        "ASD_mean±std",
    ]
    (out_dir / "detailed_dsc_hd95_nsd_asd_table.md").write_text(
        detail[compact_cols].to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary_dsc_hd95_nsd_asd.md").write_text(
        summary.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


def patch_cuda_to_cpu() -> None:
    torch.Tensor.cuda = lambda self, *args, **kwargs: self
    torch.nn.Module.cuda = lambda self, *args, **kwargs: self


def run_inference(name: str, device: str, max_cases: int | None) -> np.ndarray:
    if str(GACPS_ROOT) not in sys.path:
        sys.path.insert(0, str(GACPS_ROOT))

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    if device == "cpu":
        patch_cuda_to_cpu()

    from networks.magicnet import VNet_Magic
    from networks.vnet import VNet
    from utils import test_amos, test_util_vnet_AB

    cfg = EXPERIMENTS[name]
    test_list = read_test_list(cfg)
    if max_cases is not None:
        test_list = test_list[:max_cases]

    patch_size = (96, 96, 96)
    if cfg["type"] == "cps":
        model_a = VNet(n_channels=1, n_classes=cfg["num_classes"]).cuda().eval()
        model_b = VNet(n_channels=1, n_classes=cfg["num_classes"]).cuda().eval()
        model_a.load_state_dict(torch.load(resolve_path(cfg, "ckpt_a"), map_location=device))
        model_b.load_state_dict(torch.load(resolve_path(cfg, "ckpt_b"), map_location=device))
        _, _, metric_final = test_util_vnet_AB.validation_all_case(
            model_a,
            model_b,
            cfg["num_classes"],
            str(cfg["base_dir"]),
            test_list,
            patch_size,
            stride_xy=32,
            stride_z=16,
        )
        return metric_final

    model = VNet_Magic(n_channels=1, n_classes=cfg["num_classes"]).cuda().eval()
    model.load_state_dict(torch.load(resolve_path(cfg, "ckpt"), map_location=device))
    _, _, metric_final = test_amos.validation_all_case(
        model,
        cfg["num_classes"],
        str(cfg["base_dir"]),
        test_list,
        patch_size,
        stride_xy=32,
        stride_z=16,
    )
    return metric_final


def resolve_path(cfg: dict, key: str) -> Path:
    path = Path(cfg[key])
    if path.exists():
        return path
    fallback = cfg.get(f"{key}_fallback")
    if fallback is not None and Path(fallback).exists():
        return Path(fallback)
    raise FileNotFoundError(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=[*EXPERIMENTS.keys(), "all"], default="all")
    parser.add_argument("--out-dir", default=str(ROOT / "eval_outputs"))
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--max-cases", type=int, default=None)
    args = parser.parse_args()

    names = list(EXPERIMENTS) if args.exp == "all" else [args.exp]
    metrics_by_exp = {}
    for name in names:
        cfg = EXPERIMENTS[name]
        if args.run_inference:
            metric_final = run_inference(name, args.device, args.max_cases)
            suffix = f"{name}_rerun_metric_final.npy"
            if args.max_cases is not None:
                suffix = f"{name}_rerun_first{args.max_cases}_metric_final.npy"
            out_metric = Path(args.out_dir) / suffix
            out_metric.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_metric, metric_final)
            print(f"Saved rerun metrics: {out_metric}")
        else:
            metric_final = np.load(resolve_path(cfg, "metric_npy"), allow_pickle=True)
        metrics_by_exp[name] = metric_final

    write_tables(metrics_by_exp, Path(args.out_dir))


if __name__ == "__main__":
    main()
