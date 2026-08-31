#!/usr/bin/env python3

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lib.datasets.utils import get_train_and_val_datasets
from lib.models.my_arcface import MyArcFace


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_VERSION = "v1.1.0_10d_70_30"

CHECKPOINTS = {
    "Baseline": "saved_models/s21_glint360k_r50_512d_gmdb_10d_v1.1.0_10d_70_30_bs64_size112_channels3_e6.pt",
    "PD25": "saved_models/s22_glint360k_r50_512d_gmdb_10d_v1.1.0_10d_70_30_pd25_bs64_size112_channels3_e10.pt",
    "PD50": "saved_models/s23_glint360k_r50_512d_gmdb_10d_v1.1.0_10d_70_30_pd50_bs64_size112_channels3_e34.pt",
    "PD75": "saved_models/s24_glint360k_r50_512d_gmdb_10d_v1.1.0_10d_70_30_pd75_bs64_size112_channels3_e29.pt",
    "PD100": "saved_models/s25_glint360k_r50_512d_gmdb_10d_v1.1.0_10d_70_30_pd100_bs64_size112_channels3_e20.pt",
}

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
        total_top3 += top3.eq(targets.view(-1, 1)).any(dim=1).sum().item()
        total_top5 += top5.eq(targets.view(-1, 1)).any(dim=1).sum().item()
        total_n += targets.size(0)

        for target, pred in zip(targets.cpu().tolist(), preds.cpu().tolist()):
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
            "accuracy": correct[disease] / total[disease]
        })

    df = pd.DataFrame(rows)

    overall = total_correct / total_n
    overall_top3 = total_top3 / total_n
    overall_top5 = total_top5 / total_n
    mean_class = df["accuracy"].mean()

    return df, overall, mean_class, overall_top3, overall_top5


def main():

    print("=" * 100)
    print("GESTALTMATCHER-ARC V2 PER-DISEASE EVALUATION")
    print("=" * 100)

    train, val = get_train_and_val_datasets(
        "gmdb",
        "10d",
        BASE_VERSION,
        112,
        3,
        "data",
        img_postfix="_aligned"
    )

    # Critical: no random augmentation during evaluation
    val.augment = False

    assert len(val) == 558
    assert val.get_num_classes() == 10

    lookup = val.get_lookup_table()

    print("Device:", DEVICE)
    print("Validation images:", len(val))
    print("Lookup:", lookup)

    loader = DataLoader(
        val,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    master = None
    summary = []

    for condition, checkpoint in CHECKPOINTS.items():

        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)

        print("\n" + "=" * 100)
        print(condition)
        print("=" * 100)

        model = load_model(checkpoint, len(lookup))

        per_class, overall, mean_class, overall_top3, overall_top5 = evaluate(
            model,
            loader,
            lookup
        )

        print(f"Overall Top-1: {overall:.4%}")
        print(f"Mean Top-1:    {mean_class:.4%}")
        print(f"Overall Top-3: {overall_top3:.4%}")
        print(f"Overall Top-5: {overall_top5:.4%}")

        show = per_class.copy()
        show["accuracy"] *= 100

        print(show.to_string(
            index=False,
            formatters={"accuracy": lambda x: f"{x:.2f}%"}
        ))

        summary.append({
            "condition": condition,
            "overall_top1": overall,
            "mean_top1": mean_class,
            "overall_top3": overall_top3,
            "overall_top5": overall_top5
        })

        temp = per_class[["disease", "test_n", "accuracy"]].copy()
        temp = temp.rename(columns={"accuracy": condition})

        if master is None:
            master = temp
        else:
            master = master.merge(
                temp[["disease", condition]],
                on="disease"
            )

    master.insert(
        1,
        "real_train_n",
        master["disease"].map(REAL_TRAIN_COUNTS)
    )

    for condition in ["PD25", "PD50", "PD75", "PD100"]:
        master[f"{condition}_delta_pp"] = (
            master[condition] - master["Baseline"]
        ) * 100

    master = master.sort_values(
        "real_train_n",
        ascending=False
    ).reset_index(drop=True)

    Path("analysis_v2").mkdir(exist_ok=True)

    master.to_csv(
        "analysis_v2/arc_v2_per_disease_accuracy.csv",
        index=False
    )

    pd.DataFrame(summary).to_csv(
        "analysis_v2/arc_v2_overall_summary.csv",
        index=False
    )

    print("\n" + "=" * 100)
    print("FINAL COMPARISON")
    print("=" * 100)

    display = master.copy()

    for col in ["Baseline", "PD25", "PD50", "PD75", "PD100"]:
        display[col] *= 100

    print(display.to_string(index=False))

    print("\nSaved:")
    print("analysis_v2/arc_v2_per_disease_accuracy.csv")
    print("analysis_v2/arc_v2_overall_summary.csv")


if __name__ == "__main__":
    main()
