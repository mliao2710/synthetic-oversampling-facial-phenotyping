#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
from datasets import load_dataset

HF_NAME = "aaronwzl/gmdb_ten_disease_subset"
VERSION = "v1.1.0_10d_70_30"
OUT = Path("data/GestaltMatcherDB") / VERSION / "gmdb_metadata"

print("Loading canonical 10-disease dataset...")
ds = load_dataset(HF_NAME)

train = ds["train"].to_pandas()
test = ds["test"].to_pandas()
full = pd.concat([train, test], ignore_index=True)

# Safety checks
assert len(train) == 1289
assert len(test) == 558
assert full["label"].nunique() == 10

assert len(
    set(train["image_id"].astype(str)) &
    set(test["image_id"].astype(str))
) == 0

assert len(
    set(train["patient_id"].astype(str)) &
    set(test["patient_id"].astype(str))
) == 0

OUT.mkdir(parents=True, exist_ok=True)

# Use the disease-name strings expected by GestaltMatcher-Arc.
# internal_syndrome_name corresponds to the original GMDB naming convention.
train_arc = pd.DataFrame({
    "image_id": train["image_id"].astype(str),
    "label": train["internal_syndrome_name"]
})

test_arc = pd.DataFrame({
    "image_id": test["image_id"].astype(str),
    "label": test["internal_syndrome_name"]
})

# Full metadata
full.to_csv(
    OUT / "gmdb_metadata.csv",
    index=False
)

train_arc.to_csv(
    OUT / f"gmdb_train_images_{VERSION}.csv",
    index=False
)

test_arc.to_csv(
    OUT / f"gmdb_val_images_{VERSION}.csv",
    index=False
)

print("\n" + "="*80)
print("GESTALTMATCHER-ARC V2 DATASET CREATED")
print("="*80)
print("Version:", VERSION)
print("Train images:", len(train_arc))
print("Test images:", len(test_arc))
print("Train patients:", train["patient_id"].nunique())
print("Test patients:", test["patient_id"].nunique())
print("Diseases:", train_arc["label"].nunique())

print("\nTrain distribution:")
print(train_arc["label"].value_counts().to_string())

print("\nTest distribution:")
print(test_arc["label"].value_counts().to_string())

print("\nFiles:")
for p in sorted(OUT.iterdir()):
    print(" ", p)
