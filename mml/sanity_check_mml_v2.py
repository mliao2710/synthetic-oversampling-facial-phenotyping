from pathlib import Path
import pandas as pd
import torch

RUNS = {
    "Baseline": Path("runs_10d/v1.1.0_10d_70_30_seed11_20260807_125218"),
    "PD25": Path("runs_10d/v1.1.0_10d_70_30_pd25_seed11_20260809_184421"),
    "PD50": Path("runs_10d/v1.1.0_10d_70_30_pd50_seed11_20260809_184421"),
    "PD75": Path("runs_10d/v1.1.0_10d_70_30_pd75_seed11_20260809_184421"),
    "PD100": Path("runs_10d/v1.1.0_10d_70_30_pd100_seed11_20260809_184421"),
}

EXPECTED = {
    "Baseline": (1289, 0, 2578),
    "PD25": (1289, 433, 3011),
    "PD50": (1289, 866, 3444),
    "PD75": (1289, 1298, 3876),
    "PD100": (1289, 1727, 4305),
}

print("=" * 100)
print("GESTALTMML V2 SANITY AUDIT")
print("=" * 100)

baseline_val = None
baseline_real_ids = None

for condition, run in RUNS.items():

    print("\n" + "=" * 100)
    print(condition)
    print("=" * 100)

    train = pd.read_csv(run / "train_examples.csv", dtype={"image_id": str})
    val = pd.read_csv(run / "validation_examples.csv", dtype={"image_id": str})

    summary = pd.read_json(run / "dataset_summary.json", typ="series")

    real = train[train["kind"] == "real_matched"]
    real_star = train[train["kind"] == "real_star"]
    synth = train[train["kind"] == "synthetic"]

    exp_real, exp_synth, exp_total = EXPECTED[condition]

    print("Real matched:", len(real))
    print("Real star:", len(real_star))
    print("Synthetic:", len(synth))
    print("Total training examples:", len(train))
    print("Validation:", len(val))

    assert len(real) == exp_real
    assert len(real_star) == exp_real
    assert len(synth) == exp_synth
    assert len(train) == exp_total
    assert len(val) == 558

    # Exact real-image duplication check
    assert set(real["image_id"]) == set(real_star["image_id"])

    # Training real images vs validation images
    real_val_overlap = set(real["image_id"]) & set(val["image_id"])
    print("Real train / validation image overlap:", len(real_val_overlap))
    assert len(real_val_overlap) == 0

    # Synthetic filenames themselves must not appear in validation
    synth_val_overlap = set(synth["image_id"]) & set(val["image_id"])
    print("Synthetic ID / validation ID overlap:", len(synth_val_overlap))
    assert len(synth_val_overlap) == 0

    # Validation labels
    print("Validation diseases:", val["label"].nunique())
    assert val["label"].nunique() == 10

    # Store baseline references
    if condition == "Baseline":
        baseline_val = val[["image_id", "label"]].reset_index(drop=True)
        baseline_real_ids = set(real["image_id"])
    else:
        # All conditions must use identical validation set and ordering
        check = val[["image_id", "label"]].reset_index(drop=True)
        assert check.equals(baseline_val)

        # Same 1289 unique real training images in all conditions
        assert set(real["image_id"]) == baseline_real_ids

    # Best checkpoint consistency
    ckpt = torch.load(run / "best.pt", map_location="cpu")

    print("Best checkpoint epoch:", ckpt["epoch"])
    print("Saved Top-1:", ckpt["metrics"]["top1"])
    print("Saved Mean Top-1:", ckpt["metrics"]["mean_top1"])

print("\n" + "=" * 100)
print("BASIC DATASET AUDIT PASSED")
print("=" * 100)
print("All conditions:")
print("  ✓ same 1289 real training images")
print("  ✓ same 558 validation images")
print("  ✓ zero real train/test image overlap")
print("  ✓ zero synthetic-ID/test-ID overlap")
print("  ✓ expected synthetic counts")
print("  ✓ 10 diseases")
