#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path
import glob
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.datasets.utils import get_train_and_val_datasets
from lib.models.my_arcface import MyArcFace


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_VERSION = "v1.1.0_10d_aaron70_30"
CONDITIONS = ["Baseline", "PD25", "PD50", "PD75", "PD100"]
SEEDS = [11, 22, 33]

REAL_TRAIN_COUNTS = {
    "22q11.2 deletion syndrome": 41,
    "Angelman syndrome": 111,
    "Cornelia de Lange syndrome": 312,
    "KBG syndrome": 125,
    "Kabuki syndrome": 144,
    "Nicolaides-Baraitser syndrome": 68,
    "Noonan syndrome": 145,
    "Rubinstein-Taybi syndrome": 81,
    "Smith-Magenis syndrome": 71,
    "Williams-Beuren syndrome": 191,
}

# Exact training logs corresponding to each run.
# Seed 11 uses the original paper runs.
# Seeds 22/33 use the new multi-seed runs.
LOGS = {
    (11, "Baseline"): "logs/gm10d_v2_base_16573601.stdout",
    (11, "PD25"):     "logs/gm10d_v2_pd25_16664242.stdout",
    (11, "PD50"):     "logs/gm10d_v2_pd50_16664448.stdout",
    (11, "PD75"):     "logs/gm10d_v2_pd75_16664244.stdout",
    (11, "PD100"):    "logs/gm10d_v2_pd100_16664245.stdout",

    (22, "Baseline"): "logs/ms_base_s22_24632915.stdout",
    (22, "PD25"):     "logs/ms_pd25_s22_24632916.stdout",
    (22, "PD50"):     "logs/ms_pd50_s22_24632917.stdout",
    (22, "PD75"):     "logs/ms_pd75_s22_24632918.stdout",
    (22, "PD100"):    "logs/ms_pd100_s22_24632919.stdout",

    (33, "Baseline"): "logs/ms_base_s33_24632920.stdout",
    (33, "PD25"):     "logs/ms_pd25_s33_24632921.stdout",
    (33, "PD50"):     "logs/ms_pd50_s33_24632922.stdout",
    (33, "PD75"):     "logs/ms_pd75_s33_24632923.stdout",
    (33, "PD100"):    "logs/ms_pd100_s33_24632924.stdout",
}


# ============================================================
# CHECKPOINT SELECTION
# ============================================================

def select_best_mean_top1(log_path):
    """
    Select the checkpoint with the highest overall Top-1 accuracy.
    Ties are resolved in favor of the earlier epoch.

    Returns:
        epoch
        training-log overall Top-1
        training-log Mean Top-1
        checkpoint path
    """

    with open(log_path) as f:
        log = f.read()

    pattern = re.compile(
        r"Average BCE Loss \(([\d.eE+-]+)\) during validation"
        r".*?Top-1 accuracy: ([\d.eE+-]+), Top-5 accuracy:"
        r".*?Mean Top-1 accuracy: ([\d.eE+-]+),"
        r".*?Saving model in:\s*(\S+?\.pt)",
        re.S
    )

    records = []

    for loss, top1, mean1, checkpoint in pattern.findall(log):
        epoch_match = re.search(r"_e(\d+)\.pt$", checkpoint)

        if not epoch_match:
            continue

        epoch = int(epoch_match.group(1))

        records.append({
            "epoch": epoch,
            "overall_top1_log": float(top1),
            "mean_top1_log": float(mean1),
            "loss": float(loss),
            "checkpoint": checkpoint,
        })

    if len(records) != 50:
        raise RuntimeError(
            f"{log_path}: expected 50 parsed epochs, found {len(records)}"
        )

    # Highest Mean Top-1; ties -> earliest epoch
    best = max(
        records,
        key=lambda r: (r["overall_top1_log"], -r["epoch"])
    )

    checkpoint = Path(best["checkpoint"])

    # Training output may print only filename rather than saved_models/filename
    if not checkpoint.is_file():
        checkpoint = Path("saved_models") / checkpoint.name

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Selected checkpoint not found: {checkpoint}"
        )

    best["checkpoint"] = str(checkpoint)

    return best


# ============================================================
# MODEL / EVALUATION
# ============================================================

def load_model(path, n_classes):

    model = MyArcFace(
        n_classes,
        dataset_base="saved_models/glint360k_r50.onnx",
        device=DEVICE,
        freeze=True
    ).to(DEVICE)

    state = torch.load(path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def evaluate(model, loader, lookup):

    correct = defaultdict(int)
    total = defaultdict(int)

    total_correct = 0
    total_top3 = 0
    total_top5 = 0
    total_n = 0

    for images, targets in loader:

        images = images.to(DEVICE, dtype=torch.float32)
        targets = targets.to(DEVICE, dtype=torch.int64)

        logits, _ = model(images)

        preds = logits.argmax(dim=1)
        top3 = logits.topk(k=3, dim=1).indices
        top5 = logits.topk(k=5, dim=1).indices

        total_correct += (preds == targets).sum().item()

        total_top3 += (
            top3.eq(targets.view(-1, 1))
            .any(dim=1)
            .sum()
            .item()
        )

        total_top5 += (
            top5.eq(targets.view(-1, 1))
            .any(dim=1)
            .sum()
            .item()
        )

        total_n += targets.size(0)

        for target, pred in zip(
            targets.cpu().tolist(),
            preds.cpu().tolist()
        ):

            disease = lookup[target]

            total[disease] += 1

            if target == pred:
                correct[disease] += 1

    rows = []

    for disease in lookup:

        rows.append({
            "disease": disease,
            "test_n": total[disease],
            "correct": correct[disease],
            "accuracy": correct[disease] / total[disease],
        })

    df = pd.DataFrame(rows)

    overall_top1 = total_correct / total_n
    overall_top3 = total_top3 / total_n
    overall_top5 = total_top5 / total_n
    mean_top1 = df["accuracy"].mean()

    return (
        df,
        overall_top1,
        mean_top1,
        overall_top3,
        overall_top5
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 110)
    print("GESTALTMATCHER-ARC FINAL MULTI-SEED EVALUATION")
    print("Checkpoint rule: maximum overall Top-1; ties -> earliest epoch")
    print("=" * 110)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train, val = get_train_and_val_datasets(
        "gmdb",
        "10d",
        BASE_VERSION,
        112,
        3,
        "data",
        img_postfix="_aligned"
    )

    # No random augmentation at evaluation
    val.augment = False

    assert len(val) == 558
    assert val.get_num_classes() == 10

    lookup = val.get_lookup_table()

    print("\nDevice:", DEVICE)
    print("Evaluation images:", len(val))
    print("Classes:", len(lookup))
    print("Lookup:", lookup)

    loader = DataLoader(
        val,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    outdir = Path("analysis_multiseed_final")
    outdir.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Select all 15 checkpoints first
    # --------------------------------------------------------

    selections = []

    print("\n" + "=" * 110)
    print("CHECKPOINT SELECTION")
    print("=" * 110)

    for seed in SEEDS:

        for condition in CONDITIONS:

            log_path = LOGS[(seed, condition)]

            if not Path(log_path).is_file():
                raise FileNotFoundError(log_path)

            best = select_best_mean_top1(log_path)

            row = {
                "seed": seed,
                "condition": condition,
                "epoch": best["epoch"],
                "log_overall_top1": best["overall_top1_log"],
                "log_mean_top1": best["mean_top1_log"],
                "validation_loss": best["loss"],
                "checkpoint": best["checkpoint"],
                "log_file": log_path,
            }

            selections.append(row)

            print(
                f"Seed {seed:2d} | "
                f"{condition:8s} | "
                f"epoch {best['epoch']:2d} | "
                f"Mean Top-1 {best['mean_top1_log']*100:7.3f}% | "
                f"Overall Top-1 {best['overall_top1_log']*100:7.3f}%"
            )

    selections_df = pd.DataFrame(selections)

    selections_df.to_csv(
        outdir / "selected_checkpoints.csv",
        index=False
    )

    # --------------------------------------------------------
    # Evaluate all 15 selected checkpoints
    # --------------------------------------------------------

    overall_rows = []
    disease_rows = []

    print("\n" + "=" * 110)
    print("FINAL EVALUATION")
    print("=" * 110)

    for row in selections:

        seed = row["seed"]
        condition = row["condition"]
        epoch = row["epoch"]
        checkpoint = row["checkpoint"]

        print("\n" + "-" * 110)
        print(
            f"Seed {seed} | {condition} | "
            f"selected epoch {epoch}"
        )
        print("-" * 110)

        model = load_model(checkpoint, len(lookup))

        (
            per_class,
            overall_top1,
            mean_top1,
            overall_top3,
            overall_top5
        ) = evaluate(
            model,
            loader,
            lookup
        )

        print(f"Overall Top-1: {overall_top1:.4%}")
        print(f"Mean Top-1:    {mean_top1:.4%}")
        print(f"Overall Top-3: {overall_top3:.4%}")
        print(f"Overall Top-5: {overall_top5:.4%}")

        overall_rows.append({
            "seed": seed,
            "condition": condition,
            "selected_epoch": epoch,
            "overall_top1": overall_top1,
            "mean_top1": mean_top1,
            "overall_top3": overall_top3,
            "overall_top5": overall_top5,
            "checkpoint": checkpoint,
        })

        for _, r in per_class.iterrows():

            disease_rows.append({
                "seed": seed,
                "condition": condition,
                "selected_epoch": epoch,
                "disease": r["disease"],
                "real_train_n": REAL_TRAIN_COUNTS[r["disease"]],
                "test_n": int(r["test_n"]),
                "correct": int(r["correct"]),
                "accuracy": r["accuracy"],
            })

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    overall_raw = pd.DataFrame(overall_rows)
    disease_raw = pd.DataFrame(disease_rows)

    overall_raw.to_csv(
        outdir / "overall_raw_15runs.csv",
        index=False
    )

    disease_raw.to_csv(
        outdir / "per_disease_raw_15runs.csv",
        index=False
    )

    # --------------------------------------------------------
    # Aggregate overall metrics: mean ± sample SD
    # --------------------------------------------------------

    metrics = [
        "overall_top1",
        "mean_top1",
        "overall_top3",
        "overall_top5"
    ]

    overall_summary = (
        overall_raw
        .groupby("condition")[metrics]
        .agg(["mean", "std"])
        .reindex(CONDITIONS)
    )

    # Flatten MultiIndex columns
    overall_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in overall_summary.columns
    ]

    overall_summary = overall_summary.reset_index()

    overall_summary.to_csv(
        outdir / "overall_mean_sd.csv",
        index=False
    )

    # --------------------------------------------------------
    # Paired seed-wise deltas vs Baseline
    # --------------------------------------------------------

    delta_rows = []

    for seed in SEEDS:

        seed_df = overall_raw[
            overall_raw["seed"] == seed
        ].set_index("condition")

        for condition in CONDITIONS[1:]:

            d = {
                "seed": seed,
                "condition": condition,
            }

            for metric in metrics:
                d[f"{metric}_delta_pp"] = (
                    seed_df.loc[condition, metric]
                    - seed_df.loc["Baseline", metric]
                ) * 100

            delta_rows.append(d)

    overall_deltas_raw = pd.DataFrame(delta_rows)

    overall_deltas_raw.to_csv(
        outdir / "overall_deltas_vs_baseline_raw.csv",
        index=False
    )

    delta_metrics = [
        c for c in overall_deltas_raw.columns
        if c.endswith("_delta_pp")
    ]

    overall_delta_summary = (
        overall_deltas_raw
        .groupby("condition")[delta_metrics]
        .agg(["mean", "std"])
        .reindex(CONDITIONS[1:])
    )

    overall_delta_summary.columns = [
        f"{metric}_{stat}"
        for metric, stat in overall_delta_summary.columns
    ]

    overall_delta_summary = overall_delta_summary.reset_index()

    overall_delta_summary.to_csv(
        outdir / "overall_deltas_vs_baseline_mean_sd.csv",
        index=False
    )

    # --------------------------------------------------------
    # Per-disease mean ± SD
    # --------------------------------------------------------

    disease_summary = (
        disease_raw
        .groupby(
            [
                "disease",
                "real_train_n",
                "test_n",
                "condition"
            ]
        )["accuracy"]
        .agg(["mean", "std"])
        .reset_index()
    )

    disease_summary = disease_summary.rename(
        columns={
            "mean": "accuracy_mean",
            "std": "accuracy_sd"
        }
    )

    disease_summary.to_csv(
        outdir / "per_disease_mean_sd.csv",
        index=False
    )

    # --------------------------------------------------------
    # Per-disease paired deltas vs Baseline
    # --------------------------------------------------------

    disease_delta_rows = []

    for seed in SEEDS:

        seed_df = disease_raw[
            disease_raw["seed"] == seed
        ]

        baseline = (
            seed_df[
                seed_df["condition"] == "Baseline"
            ]
            .set_index("disease")["accuracy"]
        )

        for condition in CONDITIONS[1:]:

            current = (
                seed_df[
                    seed_df["condition"] == condition
                ]
                .set_index("disease")["accuracy"]
            )

            for disease in lookup:

                disease_delta_rows.append({
                    "seed": seed,
                    "condition": condition,
                    "disease": disease,
                    "real_train_n": REAL_TRAIN_COUNTS[disease],
                    "test_n": int(
                        seed_df[
                            seed_df["disease"] == disease
                        ]["test_n"].iloc[0]
                    ),
                    "delta_pp": (
                        current[disease]
                        - baseline[disease]
                    ) * 100,
                })

    disease_deltas_raw = pd.DataFrame(
        disease_delta_rows
    )

    disease_deltas_raw.to_csv(
        outdir / "per_disease_deltas_vs_baseline_raw.csv",
        index=False
    )

    disease_delta_summary = (
        disease_deltas_raw
        .groupby(
            [
                "disease",
                "real_train_n",
                "test_n",
                "condition"
            ]
        )["delta_pp"]
        .agg(["mean", "std"])
        .reset_index()
    )

    disease_delta_summary = disease_delta_summary.rename(
        columns={
            "mean": "delta_pp_mean",
            "std": "delta_pp_sd"
        }
    )

    disease_delta_summary.to_csv(
        outdir / "per_disease_deltas_vs_baseline_mean_sd.csv",
        index=False
    )

    # --------------------------------------------------------
    # Manuscript-friendly wide disease table
    # --------------------------------------------------------

    disease_wide_mean = disease_summary.pivot(
        index=["disease", "real_train_n", "test_n"],
        columns="condition",
        values="accuracy_mean"
    ).reset_index()

    disease_wide_sd = disease_summary.pivot(
        index=["disease", "real_train_n", "test_n"],
        columns="condition",
        values="accuracy_sd"
    ).reset_index()

    disease_wide_mean = disease_wide_mean[
        ["disease", "real_train_n", "test_n"] + CONDITIONS
    ]

    disease_wide_mean = disease_wide_mean.sort_values(
        "real_train_n",
        ascending=True
    )

    disease_wide_mean.to_csv(
        outdir / "per_disease_mean_wide.csv",
        index=False
    )

    disease_wide_sd = disease_wide_sd[
        ["disease", "real_train_n", "test_n"] + CONDITIONS
    ]

    disease_wide_sd = disease_wide_sd.sort_values(
        "real_train_n",
        ascending=True
    )

    disease_wide_sd.to_csv(
        outdir / "per_disease_sd_wide.csv",
        index=False
    )

    # --------------------------------------------------------
    # Print final manuscript-ready overall table
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("MANUSCRIPT-READY OVERALL RESULTS: MEAN ± SAMPLE SD ACROSS 3 SEEDS")
    print("=" * 110)

    for _, r in overall_summary.iterrows():

        print(
            f"{r['condition']:8s} | "
            f"Top-1 "
            f"{r['overall_top1_mean']*100:.2f} ± "
            f"{r['overall_top1_std']*100:.2f}% | "
            f"Mean Top-1 "
            f"{r['mean_top1_mean']*100:.2f} ± "
            f"{r['mean_top1_std']*100:.2f}% | "
            f"Top-3 "
            f"{r['overall_top3_mean']*100:.2f} ± "
            f"{r['overall_top3_std']*100:.2f}% | "
            f"Top-5 "
            f"{r['overall_top5_mean']*100:.2f} ± "
            f"{r['overall_top5_std']*100:.2f}%"
        )

    print("\n" + "=" * 110)
    print("PAIRED DELTAS VS BASELINE: MEAN ± SAMPLE SD")
    print("=" * 110)

    for _, r in overall_delta_summary.iterrows():

        print(
            f"{r['condition']:8s} | "
            f"Top-1 Δ "
            f"{r['overall_top1_delta_pp_mean']:+.2f} ± "
            f"{r['overall_top1_delta_pp_std']:.2f} pp | "
            f"Mean Top-1 Δ "
            f"{r['mean_top1_delta_pp_mean']:+.2f} ± "
            f"{r['mean_top1_delta_pp_std']:.2f} pp"
        )

    print("\n" + "=" * 110)
    print("FILES SAVED")
    print("=" * 110)

    for path in sorted(outdir.glob("*.csv")):
        print(path)

    print("\nFINAL MULTI-SEED EVALUATION COMPLETE.")


if __name__ == "__main__":
    main()
