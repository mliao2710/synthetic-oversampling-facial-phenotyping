from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import ViltForQuestionAnswering, ViltProcessor

# Import the EXACT dataset/preprocessing code used during training
from train_gestaltmml_10d import (
    DS,
    collate,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "dandelin/vilt-b32-mlm"

RUNS = {
    "Baseline": Path(
        "runs_10d/v1.1.0_10d_70_30_seed11_20260807_125218"
    ),
    "PD25": Path(
        "runs_10d/v1.1.0_10d_70_30_pd25_seed11_20260809_184421"
    ),
    "PD50": Path(
        "runs_10d/v1.1.0_10d_70_30_pd50_seed11_20260809_184421"
    ),
    "PD75": Path(
        "runs_10d/v1.1.0_10d_70_30_pd75_seed11_20260809_184421"
    ),
    "PD100": Path(
        "runs_10d/v1.1.0_10d_70_30_pd100_seed11_20260809_184421"
    ),
}

OUT = Path("analysis_v2")
OUT.mkdir(exist_ok=True)


def load_validation(run):
    p = run / "validation_examples.csv"
    df = pd.read_csv(p)

    required = {"image_id", "image_path", "text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"{p} missing columns: {missing}")

    examples = df.to_dict("records")
    return df, examples


def load_model(run):
    ckpt_path = run / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu")

    label2id = ckpt["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    # Use processor saved with this best checkpoint
    proc_dir = run / "processor"
    if proc_dir.exists():
        proc = ViltProcessor.from_pretrained(proc_dir)
    else:
        proc = ViltProcessor.from_pretrained(MODEL_NAME)

    model = ViltForQuestionAnswering.from_pretrained(
        MODEL_NAME,
        num_labels=10,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, proc, label2id, ckpt


@torch.inference_mode()
def evaluate_per_disease(run):
    val_df, examples = load_validation(run)
    model, proc, label2id, ckpt = load_model(run)

    ds = DS(examples, proc, label2id, 40)

    loader = DataLoader(
        ds,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate(proc),
    )

    correct = defaultdict(int)
    total = defaultdict(int)

    overall_correct = 0
    overall_total = 0

    offset = 0

    for batch in loader:
        labels = batch.pop("labels").argmax(1)

        x = {
            k: v.to(DEVICE, non_blocking=True)
            for k, v in batch.items()
        }

        logits = model(**x).logits
        preds = logits.argmax(1).cpu()

        bs = len(labels)

        # Use validation CSV labels as an additional sanity check
        for i in range(bs):
            disease = str(val_df.iloc[offset + i]["label"])

            expected = label2id[disease]
            actual = int(labels[i])

            if expected != actual:
                raise RuntimeError(
                    f"Label mismatch for {disease}: "
                    f"CSV={expected}, dataset={actual}"
                )

            total[disease] += 1

            is_correct = int(preds[i]) == actual
            correct[disease] += int(is_correct)

            overall_correct += int(is_correct)
            overall_total += 1

        offset += bs

    rows = []

    for disease in sorted(total):
        rows.append({
            "disease": disease,
            "test_n": total[disease],
            "correct_n": correct[disease],
            "accuracy": correct[disease] / total[disease],
        })

    result = pd.DataFrame(rows)

    overall = overall_correct / overall_total
    macro = result["accuracy"].mean()

    print()
    print("Run:", run)
    print("Best epoch:", ckpt["epoch"])
    print("Checkpoint Top-1:", ckpt["metrics"]["top1"])
    print("Recomputed Top-1:", overall)
    print("Checkpoint Mean Top-1:", ckpt["metrics"]["mean_top1"])
    print("Recomputed Mean Top-1:", macro)
    print()

    # CRITICAL integrity check
    if not np.isclose(
        overall,
        float(ckpt["metrics"]["top1"]),
        atol=1e-8,
    ):
        raise RuntimeError(
            "Recomputed overall Top-1 does not match checkpoint!"
        )

    if not np.isclose(
        macro,
        float(ckpt["metrics"]["mean_top1"]),
        atol=1e-8,
    ):
        raise RuntimeError(
            "Recomputed Mean Top-1 does not match checkpoint!"
        )

    return result, overall, macro


# --------------------------------------------------
# Evaluate all five conditions
# --------------------------------------------------

results = {}
overall_rows = []

for condition, run in RUNS.items():
    print("=" * 80)
    print(condition)
    print("=" * 80)

    result, overall, macro = evaluate_per_disease(run)

    results[condition] = result

    overall_rows.append({
        "condition": condition,
        "overall_top1": overall,
        "mean_top1": macro,
    })


# --------------------------------------------------
# Verify identical validation sets
# --------------------------------------------------

base_ids = pd.read_csv(
    RUNS["Baseline"] / "validation_examples.csv"
)["image_id"].astype(str).tolist()

for condition, run in RUNS.items():
    ids = pd.read_csv(
        run / "validation_examples.csv"
    )["image_id"].astype(str).tolist()

    if ids != base_ids:
        raise RuntimeError(
            f"Validation set/order differs for {condition}"
        )

print("\nValidation sets verified identical across all conditions.")


# --------------------------------------------------
# Combine per-disease results
# --------------------------------------------------

combined = None

for condition, df in results.items():
    x = df[["disease", "test_n", "accuracy"]].copy()

    x = x.rename(columns={
        "accuracy": condition,
        "test_n": f"{condition}_test_n",
    })

    if combined is None:
        combined = x
    else:
        combined = combined.merge(
            x,
            on="disease",
            how="outer",
        )


# Test N should be identical, so retain one column
combined["test_n"] = combined["Baseline_test_n"]

for condition in ["PD25", "PD50", "PD75", "PD100"]:
    if not (
        combined[f"{condition}_test_n"]
        == combined["Baseline_test_n"]
    ).all():
        raise RuntimeError(
            f"Test counts differ for {condition}"
        )


# --------------------------------------------------
# Add REAL TRAIN counts
# --------------------------------------------------

base_train = pd.read_csv(
    RUNS["Baseline"] / "train_examples.csv"
)

real = base_train[
    base_train["kind"] == "real_matched"
]

real_counts = (
    real.groupby("label")
    .size()
    .rename("real_train_n")
    .reset_index()
    .rename(columns={"label": "disease"})
)

combined = combined.merge(
    real_counts,
    on="disease",
    how="left",
)


# --------------------------------------------------
# Add SYNTHETIC counts for each condition
# --------------------------------------------------

for condition in ["PD25", "PD50", "PD75", "PD100"]:

    train = pd.read_csv(
        RUNS[condition] / "train_examples.csv"
    )

    synth = train[
        train["kind"] == "synthetic"
    ]

    counts = (
        synth.groupby("label")
        .size()
        .rename(f"{condition}_synthetic_n")
        .reset_index()
        .rename(columns={"label": "disease"})
    )

    combined = combined.merge(
        counts,
        on="disease",
        how="left",
    )

combined = combined.fillna({
    "PD25_synthetic_n": 0,
    "PD50_synthetic_n": 0,
    "PD75_synthetic_n": 0,
    "PD100_synthetic_n": 0,
})


# --------------------------------------------------
# Delta vs baseline in percentage points
# --------------------------------------------------

for condition in ["PD25", "PD50", "PD75", "PD100"]:
    combined[f"{condition}_delta_pp"] = (
        combined[condition] - combined["Baseline"]
    ) * 100


# --------------------------------------------------
# Order by real training size, largest -> smallest
# --------------------------------------------------

combined = combined.sort_values(
    "real_train_n",
    ascending=False,
)


# --------------------------------------------------
# Pretty console output
# --------------------------------------------------

display = combined[[
    "disease",
    "real_train_n",
    "PD25_synthetic_n",
    "PD50_synthetic_n",
    "PD75_synthetic_n",
    "PD100_synthetic_n",
    "test_n",
    "Baseline",
    "PD25",
    "PD50",
    "PD75",
    "PD100",
    "PD25_delta_pp",
    "PD50_delta_pp",
    "PD75_delta_pp",
    "PD100_delta_pp",
]].copy()

for c in ["Baseline", "PD25", "PD50", "PD75", "PD100"]:
    display[c] *= 100

print("\n" + "=" * 120)
print("GESTALTMML V2 PER-DISEASE RESULTS")
print("=" * 120)

print(display.to_string(index=False))


# --------------------------------------------------
# Save publication/masterdoc CSVs
# --------------------------------------------------

combined.to_csv(
    OUT / "mml_v2_per_disease_accuracy.csv",
    index=False,
)

pd.DataFrame(overall_rows).to_csv(
    OUT / "mml_v2_overall_summary.csv",
    index=False,
)

print("\nSaved:")
print(OUT / "mml_v2_per_disease_accuracy.csv")
print(OUT / "mml_v2_overall_summary.csv")
