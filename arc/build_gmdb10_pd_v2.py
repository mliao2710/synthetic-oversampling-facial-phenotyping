#!/usr/bin/env python3

from pathlib import Path
import os
import shutil
import pandas as pd
from datasets import load_dataset

HF_NAME = "aaronwzl/gmdb_ten_disease_subset"

DB_ROOT = Path("data/GestaltMatcherDB")
BASE_VERSION = "v1.1.0_10d_70_30"

PD_SHARED = DB_ROOT / "_pdidb_shared_v1.1.0_10d" / "gmdb_align"

MANIFEST_PATH = (
    DB_ROOT
    / "v1.1.0_10d_pd100"
    / "gmdb_metadata"
    / "pdidb_manifest_v1.1.0_10d_pd100.csv"
)

CONDITIONS = {
    "pd25": "selected_pd25",
    "pd50": "selected_pd50",
    "pd75": "selected_pd75",
    "pd100": "selected_pd100",
}

EXPECTED_SYNTH = {
    "pd25": 433,
    "pd50": 866,
    "pd75": 1298,
    "pd100": 1727,
}

# Canonical disease names.
DISEASE_MAP = {
    "22q": "22q11.2 deletion syndrome",
    "AS": "Angelman syndrome",
    "CdLS": "Cornelia de Lange syndrome",
    "KBGS": "KBG syndrome",
    "KS": "Kabuki syndrome",
    "NCBRS": "Nicolaides-Baraitser syndrome",
    "NS": "Noonan syndrome",
    "RTS": "Rubinstein-Taybi syndrome",
    "SMS": "Smith-Magenis syndrome",
    "WBS": "Williams-Beuren syndrome",
}


def ensure_symlink(link_path: Path, target: Path):
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()

    elif link_path.exists():
        raise RuntimeError(
            f"{link_path} exists and is not the expected symlink."
        )

    os.symlink(target.resolve(), link_path)


print("Loading canonical dataset...")
ds = load_dataset(HF_NAME)

real_train = ds["train"].to_pandas()
real_test = ds["test"].to_pandas()

assert len(real_train) == 1289
assert len(real_test) == 558
assert real_train["disease"].nunique() == 10
assert real_test["disease"].nunique() == 10

# Verify canonical train/evaluation split remains patient-disjoint.
assert not (
    set(real_train["patient_id"].astype(str))
    & set(real_test["patient_id"].astype(str))
)

assert not (
    set(real_train["image_id"].astype(str))
    & set(real_test["image_id"].astype(str))
)

manifest = pd.read_csv(MANIFEST_PATH)

assert len(manifest) == 1727
assert set(manifest["disease_abbr"].unique()) == set(DISEASE_MAP.keys())

# Baseline real CSVs
real_train_arc = pd.DataFrame({
    "image_id": real_train["image_id"].astype(str),
    "label": real_train["disease"],
})

real_test_arc = pd.DataFrame({
    "image_id": real_test["image_id"].astype(str),
    "label": real_test["disease"],
})

# Validate real label universe.
canonical_labels = set(DISEASE_MAP.values())

assert set(real_train_arc["label"]) == canonical_labels
assert set(real_test_arc["label"]) == canonical_labels

for condition, selected_col in CONDITIONS.items():

    version = f"{BASE_VERSION}_{condition}"
    out_dir = DB_ROOT / version
    meta_dir = out_dir / "gmdb_metadata"

    meta_dir.mkdir(parents=True, exist_ok=True)

    selected = manifest[
        manifest[selected_col].fillna(False).astype(bool)
    ].copy()

    assert len(selected) == EXPECTED_SYNTH[condition]

    selected["canonical_label"] = selected["disease_abbr"].map(DISEASE_MAP)

    if selected["canonical_label"].isna().any():
        raise RuntimeError(
            f"Unmapped synthetic disease label in {condition}"
        )

    synth_arc = pd.DataFrame({
        "image_id": selected["image_id"].astype(str),
        "label": selected["canonical_label"],
    })

    # Check every synthetic image physically exists.
    missing_synth = []

    for iid in synth_arc["image_id"]:
        p = PD_SHARED / f"{iid}_aligned.jpg"
        if not p.is_file():
            missing_synth.append(str(p))

    if missing_synth:
        raise FileNotFoundError(
            f"{condition}: {len(missing_synth)} synthetic images missing. "
            f"First: {missing_synth[:5]}"
        )

    # Combine the fixed real training set with synthetic data.
    combined_train = pd.concat(
        [real_train_arc, synth_arc],
        ignore_index=True
    )

    # Test set remains EXACTLY the fixed real evaluation set.
    test_arc = real_test_arc.copy()

    assert len(combined_train) == 1289 + EXPECTED_SYNTH[condition]
    assert len(test_arc) == 558

    assert combined_train["label"].nunique() == 10
    assert test_arc["label"].nunique() == 10

    assert set(combined_train["label"]) == canonical_labels
    assert set(test_arc["label"]) == canonical_labels

    # Save training/test CSVs.
    train_csv = (
        meta_dir
        / f"gmdb_train_images_{version}.csv"
    )

    val_csv = (
        meta_dir
        / f"gmdb_val_images_{version}.csv"
    )

    combined_train.to_csv(train_csv, index=False)
    test_arc.to_csv(val_csv, index=False)

    # Save canonical metadata for downstream analysis.
    full_real = pd.concat(
        [real_train, real_test],
        ignore_index=True
    )

    full_real.to_csv(
        meta_dir / "gmdb_metadata.csv",
        index=False
    )

    # Save selected synthetic manifest for reproducibility.
    selected.to_csv(
        meta_dir / f"pdidb_manifest_{version}.csv",
        index=False
    )

    # Build combined image directory using symlinks.
    align_dir = out_dir / "gmdb_align"
    align_dir.mkdir(parents=True, exist_ok=True)

    # Symlink real images.
    REAL_ROOT = Path(
        os.environ.get("GMDB_IMAGE_ROOT", "data/gmdb_crops")
    ).resolve()

    for iid in real_train_arc["image_id"]:
        src = REAL_ROOT / f"{iid}_aligned.jpg"
        dst = align_dir / f"{iid}_aligned.jpg"

        if not src.is_file():
            raise FileNotFoundError(src)

        if not dst.exists():
            os.symlink(src.resolve(), dst)

    for iid in real_test_arc["image_id"]:
        src = REAL_ROOT / f"{iid}_aligned.jpg"
        dst = align_dir / f"{iid}_aligned.jpg"

        if not src.is_file():
            raise FileNotFoundError(src)

        if not dst.exists():
            os.symlink(src.resolve(), dst)

    # Symlink synthetic images.
    for iid in synth_arc["image_id"]:
        src = PD_SHARED / f"{iid}_aligned.jpg"
        dst = align_dir / f"{iid}_aligned.jpg"

        if not dst.exists():
            os.symlink(src.resolve(), dst)

    print("\n" + "=" * 90)
    print(condition.upper())
    print("=" * 90)
    print("Version:", version)
    print("Real train:", len(real_train_arc))
    print("Synthetic:", len(synth_arc))
    print("Total train:", len(combined_train))
    print("Test:", len(test_arc))
    print("Classes:", combined_train["label"].nunique())

    print("\nSynthetic disease distribution:")
    print(
        synth_arc["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nCombined training distribution:")
    print(
        combined_train["label"]
        .value_counts()
        .sort_index()
        .to_string()
    )

print("\n" + "=" * 90)
print("ALL V2 PD DATASETS CREATED SUCCESSFULLY")
print("=" * 90)
