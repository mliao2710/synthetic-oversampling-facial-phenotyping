import os
#!/usr/bin/env python3

from pathlib import Path
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lib.models.my_arcface import MyArcFace
from lib.datasets.gestalt_matcher_dataset import GestaltMatcherDataset


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT = Path(
    os.environ.get("GESTALTMATCHER_ARC_ROOT", ".")
).resolve()

IMAGE_ROOT = Path(
    os.environ.get("GMDB_IMAGE_ROOT", "data/gmdb_crops")
).resolve()

VAL_CSV = PROJECT / (
    "data/GestaltMatcherDB/"
    "v1.1.0_10d_70_30/"
    "gmdb_metadata/"
    "gmdb_val_images_v1.1.0_10d_70_30.csv"
)

BASELINE_CKPT = PROJECT / (
    "saved_models/"
    "s21_glint360k_r50_512d_gmdb_10d_"
    "v1.1.0_10d_70_30_"
    "bs64_size112_channels3_e6.pt"
)

PD50_CKPT = PROJECT / (
    "saved_models/"
    "s23_glint360k_r50_512d_gmdb_10d_"
    "v1.1.0_10d_70_30_pd50_"
    "bs64_size112_channels3_e34.pt"
)

PRETRAINED = PROJECT / "saved_models/glint360k_r50.onnx"

OUT = PROJECT / "analysis_v2/embedding_clusters"
OUT.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
NUM_WORKERS = 4
SEED = 11


# ============================================================
# REPRODUCIBILITY
# ============================================================

np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 90)
print("GESTALTMATCHER-ARC EMBEDDING CLUSTER ANALYSIS")
print("=" * 90)
print("Device:", device)
print("Validation CSV:", VAL_CSV)
print("Image root:", IMAGE_ROOT)
print("Baseline checkpoint:", BASELINE_CKPT)
print("PD50 checkpoint:", PD50_CKPT)


# ============================================================
# BASIC FILE CHECKS
# ============================================================

for p in [
    VAL_CSV,
    BASELINE_CKPT,
    PD50_CKPT,
    PRETRAINED,
]:
    if not p.exists():
        raise FileNotFoundError(p)

val_table = pd.read_csv(VAL_CSV)

if len(val_table) != 558:
    raise RuntimeError(
        f"Expected 558 validation images; found {len(val_table)}"
    )

if val_table["label"].nunique() != 10:
    raise RuntimeError(
        f"Expected 10 diseases; found "
        f"{val_table['label'].nunique()}"
    )

print("\nValidation images:", len(val_table))
print("Diseases:", val_table["label"].nunique())


# ============================================================
# VERIFY ALL 558 IMAGES EXIST
# ============================================================

missing = []

for iid in val_table["image_id"].astype(str):
    p = IMAGE_ROOT / f"{iid}_aligned.jpg"
    if not p.is_file():
        missing.append(str(p))

print("Missing validation images:", len(missing))

if missing:
    print("\nExample missing files:")
    print("\n".join(missing[:20]))
    raise RuntimeError("Validation images are missing.")


# ============================================================
# BUILD EXACT VALIDATION DATASET
#
# augment=False is critical.
# Uses the repository's GestaltMatcherDataset preprocessing.
# ============================================================

lookup = sorted(val_table["label"].astype(str).unique())

dataset = GestaltMatcherDataset(
    imgs_dir=str(IMAGE_ROOT),
    target_file_path=str(VAL_CSV),
    in_channels=3,
    target_size=112,
    img_postfix="_aligned",
    augment=False,
    lookup_table=lookup,
    aspect_ratio=False,
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

print("\nDataset size:", len(dataset))
print("Lookup table:")
for i, disease in enumerate(lookup):
    print(i, disease)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(checkpoint_path):

    print("\nLoading:", checkpoint_path.name)

    model = MyArcFace(
        num_classes=10,
        dataset_base=str(PRETRAINED),
        device=device,
        freeze=True,
    )

    state = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    result = model.load_state_dict(
        state,
        strict=True,
    )

    print("State dict load:", result)

    model.to(device)
    model.eval()

    return model


baseline_model = load_model(BASELINE_CKPT)
pd50_model = load_model(PD50_CKPT)


# ============================================================
# EXTRACT 512-D EMBEDDINGS
# ============================================================

@torch.inference_mode()
def extract_embeddings(model, loader):

    embeddings = []
    targets = []
    predictions = []

    for data, target in loader:

        data = data.to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        target = target.to(device)

        logits, reps = model(data)

        # Sanity check
        if reps.ndim != 2 or reps.shape[1] != 512:
            raise RuntimeError(
                f"Unexpected embedding shape: {reps.shape}"
            )

        # ArcFace embeddings are commonly compared in normalized space.
        reps = F.normalize(
            reps,
            p=2,
            dim=1,
        )

        pred = logits.argmax(dim=1)

        embeddings.append(
            reps.detach().cpu().numpy()
        )
        targets.append(
            target.detach().cpu().numpy()
        )
        predictions.append(
            pred.detach().cpu().numpy()
        )

    X = np.concatenate(embeddings)
    y = np.concatenate(targets)
    pred = np.concatenate(predictions)

    return X, y, pred


print("\n" + "=" * 90)
print("EXTRACTING BASELINE EMBEDDINGS")
print("=" * 90)

X_base, y_base, pred_base = extract_embeddings(
    baseline_model,
    loader,
)

print("\n" + "=" * 90)
print("EXTRACTING PD50 EMBEDDINGS")
print("=" * 90)

X_pd50, y_pd50, pred_pd50 = extract_embeddings(
    pd50_model,
    loader,
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

assert X_base.shape == (558, 512)
assert X_pd50.shape == (558, 512)

if not np.array_equal(y_base, y_pd50):
    raise RuntimeError(
        "Baseline and PD50 target arrays differ."
    )

y = y_base

baseline_acc = np.mean(pred_base == y)
pd50_acc = np.mean(pred_pd50 == y)

print("\n" + "=" * 90)
print("CLASSIFICATION SANITY CHECK")
print("=" * 90)

print(f"Baseline Top-1: {baseline_acc*100:.2f}%")
print(f"PD50 Top-1:     {pd50_acc*100:.2f}%")

# Expected from our previous best-checkpoint analysis.
expected_baseline = 0.8817204301075269
expected_pd50 = 0.9086021505376344

print(
    "Expected Baseline:",
    f"{expected_baseline*100:.2f}%"
)

print(
    "Expected PD50:",
    f"{expected_pd50*100:.2f}%"
)

if not np.isclose(
    baseline_acc,
    expected_baseline,
    atol=1e-8,
):
    raise RuntimeError(
        "Baseline accuracy does not reproduce "
        "previous evaluation."
    )

if not np.isclose(
    pd50_acc,
    expected_pd50,
    atol=1e-8,
):
    raise RuntimeError(
        "PD50 accuracy does not reproduce "
        "previous evaluation."
    )

print("\nPASS: checkpoint predictions exactly reproduce prior results.")


# ============================================================
# GLOBAL SILHOUETTE SCORE
#
# Higher = stronger class separation.
#
# Cosine distance is appropriate for L2-normalized ArcFace
# embeddings.
# ============================================================

print("\n" + "=" * 90)
print("GLOBAL CLUSTER METRICS")
print("=" * 90)

sil_base = silhouette_score(
    X_base,
    y,
    metric="cosine",
)

sil_pd50 = silhouette_score(
    X_pd50,
    y,
    metric="cosine",
)

print(f"Baseline silhouette: {sil_base:.6f}")
print(f"PD50 silhouette:     {sil_pd50:.6f}")
print(f"Delta:               {sil_pd50 - sil_base:+.6f}")


# ============================================================
# PER-DISEASE CLUSTER METRICS
#
# Compactness:
# average cosine distance from each image to its disease centroid.
# LOWER = tighter cluster.
#
# Separation:
# distance from disease centroid to nearest other disease centroid.
# HIGHER = more isolated cluster.
#
# Ratio:
# nearest-centroid separation / compactness
# HIGHER = better.
# ============================================================

def disease_metrics(X, y, names):

    centroids = {}

    for class_id in sorted(np.unique(y)):
        Xi = X[y == class_id]

        centroid = Xi.mean(axis=0)
        centroid /= np.linalg.norm(centroid)

        centroids[class_id] = centroid

    rows = []

    for class_id in sorted(np.unique(y)):

        Xi = X[y == class_id]
        centroid = centroids[class_id]

        within = cosine_distances(
            Xi,
            centroid.reshape(1, -1),
        ).mean()

        other_distances = []

        for other_id, other_centroid in centroids.items():
            if other_id == class_id:
                continue

            d = cosine_distances(
                centroid.reshape(1, -1),
                other_centroid.reshape(1, -1),
            )[0, 0]

            other_distances.append(d)

        nearest_other = min(other_distances)

        ratio = nearest_other / within

        rows.append({
            "class_id": class_id,
            "disease": names[class_id],
            "n_test": len(Xi),
            "compactness": within,
            "nearest_centroid_separation": nearest_other,
            "separation_compactness_ratio": ratio,
        })

    return pd.DataFrame(rows)


base_metrics = disease_metrics(
    X_base,
    y,
    lookup,
)

pd50_metrics = disease_metrics(
    X_pd50,
    y,
    lookup,
)

comparison = base_metrics.merge(
    pd50_metrics,
    on=[
        "class_id",
        "disease",
        "n_test",
    ],
    suffixes=("_baseline", "_pd50"),
)

comparison["compactness_change_pct"] = (
    (
        comparison["compactness_pd50"]
        - comparison["compactness_baseline"]
    )
    / comparison["compactness_baseline"]
    * 100
)

comparison["separation_change_pct"] = (
    (
        comparison["nearest_centroid_separation_pd50"]
        - comparison["nearest_centroid_separation_baseline"]
    )
    / comparison["nearest_centroid_separation_baseline"]
    * 100
)

comparison["ratio_change_pct"] = (
    (
        comparison["separation_compactness_ratio_pd50"]
        - comparison["separation_compactness_ratio_baseline"]
    )
    / comparison["separation_compactness_ratio_baseline"]
    * 100
)


# ============================================================
# ADD REAL TRAIN COUNTS
# ============================================================

train_csv = (
    PROJECT
    / "data/GestaltMatcherDB/"
    / "v1.1.0_10d_70_30/"
    / "gmdb_metadata/"
    / "gmdb_train_images_v1.1.0_10d_70_30.csv"
)

train_df = pd.read_csv(train_csv)

real_counts = (
    train_df.groupby("label")
    .size()
    .rename("real_train_n")
    .reset_index()
    .rename(columns={"label": "disease"})
)

comparison = comparison.merge(
    real_counts,
    on="disease",
    how="left",
)

comparison = comparison.sort_values(
    "real_train_n",
    ascending=False,
)


# ============================================================
# SAVE PER-DISEASE TABLE
# ============================================================

comparison.to_csv(
    OUT / "arc_baseline_vs_pd50_cluster_metrics.csv",
    index=False,
)


# ============================================================
# SAVE RAW EMBEDDINGS
# ============================================================

np.savez_compressed(
    OUT / "arc_baseline_embeddings.npz",
    embeddings=X_base,
    labels=y,
    image_ids=val_table["image_id"].astype(str).values,
)

np.savez_compressed(
    OUT / "arc_pd50_embeddings.npz",
    embeddings=X_pd50,
    labels=y,
    image_ids=val_table["image_id"].astype(str).values,
)


# ============================================================
# SAVE SAMPLE-LEVEL METADATA
# ============================================================

sample_table = val_table.copy()

sample_table["class_id"] = y
sample_table["baseline_prediction"] = [
    lookup[i] for i in pred_base
]
sample_table["pd50_prediction"] = [
    lookup[i] for i in pred_pd50
]

sample_table.to_csv(
    OUT / "arc_embedding_samples.csv",
    index=False,
)


# ============================================================
# PCA VISUALIZATION
#
# Fit ONE reducer jointly to Baseline + PD50.
# This gives both panels the same coordinate system.
# ============================================================

combined_embeddings = np.vstack(
    [X_base, X_pd50]
)

pca = PCA(
    n_components=2,
    random_state=SEED,
)

combined_pca = pca.fit_transform(
    combined_embeddings
)

pca_base = combined_pca[:len(X_base)]
pca_pd50 = combined_pca[len(X_base):]

print("\nPCA explained variance:")
print(pca.explained_variance_ratio_)


# ============================================================
# OPTIONAL UMAP
#
# If umap-learn exists, generate UMAP too.
# Otherwise PCA results remain valid.
# ============================================================

have_umap = False

try:
    import umap

    reducer = umap.UMAP(
        n_neighbors=20,
        min_dist=0.15,
        metric="cosine",
        random_state=SEED,
    )

    combined_umap = reducer.fit_transform(
        combined_embeddings
    )

    umap_base = combined_umap[:len(X_base)]
    umap_pd50 = combined_umap[len(X_base):]

    have_umap = True

except ImportError:
    print(
        "\nUMAP not installed. "
        "Skipping UMAP visualization; PCA will still be created."
    )


# ============================================================
# COLORS
# ============================================================

cmap = plt.get_cmap("tab10")

colors = {
    i: cmap(i)
    for i in range(10)
}


# ============================================================
# PLOTTING FUNCTION
# ============================================================

def make_two_panel_plot(
    coords_base,
    coords_pd50,
    xlabel,
    ylabel,
    title,
    filename,
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7),
        sharex=True,
        sharey=True,
    )

    for ax, coords, condition in zip(
        axes,
        [coords_base, coords_pd50],
        ["Baseline", "PD50"],
    ):

        for class_id, disease in enumerate(lookup):

            mask = y == class_id

            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=24,
                alpha=0.70,
                color=colors[class_id],
                label=disease,
            )

        ax.set_title(
            condition,
            fontsize=18,
            fontweight="bold",
        )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.grid(
            alpha=0.15,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.92, 0.5),
        frameon=False,
        fontsize=10,
    )

    fig.suptitle(
        title,
        fontsize=21,
        fontweight="bold",
    )

    fig.tight_layout(
        rect=[0, 0, 0.90, 0.94]
    )

    fig.savefig(
        OUT / filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


make_two_panel_plot(
    pca_base,
    pca_pd50,
    "PC1",
    "PC2",
    "GestaltMatcher-Arc Embedding Clusters: Baseline vs. PD50",
    "arc_baseline_vs_pd50_pca.png",
)

if have_umap:

    make_two_panel_plot(
        umap_base,
        umap_pd50,
        "UMAP 1",
        "UMAP 2",
        "GestaltMatcher-Arc Embedding Clusters: Baseline vs. PD50",
        "arc_baseline_vs_pd50_umap.png",
    )


# ============================================================
# SUMMARY
# ============================================================

summary = {
    "n_test": int(len(y)),
    "embedding_dimension": int(X_base.shape[1]),
    "baseline_top1": float(baseline_acc),
    "pd50_top1": float(pd50_acc),
    "baseline_silhouette": float(sil_base),
    "pd50_silhouette": float(sil_pd50),
    "silhouette_delta": float(sil_pd50 - sil_base),
    "umap_created": bool(have_umap),
}

with open(
    OUT / "arc_embedding_summary.json",
    "w",
) as f:
    json.dump(
        summary,
        f,
        indent=2,
    )


print("\n" + "=" * 90)
print("PER-DISEASE CLUSTER COMPARISON")
print("=" * 90)

cols = [
    "disease",
    "real_train_n",
    "n_test",
    "compactness_baseline",
    "compactness_pd50",
    "compactness_change_pct",
    "nearest_centroid_separation_baseline",
    "nearest_centroid_separation_pd50",
    "separation_change_pct",
    "separation_compactness_ratio_baseline",
    "separation_compactness_ratio_pd50",
    "ratio_change_pct",
]

print(
    comparison[cols].to_string(
        index=False
    )
)


print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)

print(f"Baseline Top-1:        {baseline_acc*100:.2f}%")
print(f"PD50 Top-1:            {pd50_acc*100:.2f}%")
print()
print(f"Baseline silhouette:   {sil_base:.6f}")
print(f"PD50 silhouette:       {sil_pd50:.6f}")
print(
    f"Silhouette change:     "
    f"{sil_pd50 - sil_base:+.6f}"
)

print("\nSaved outputs:")

for p in sorted(OUT.iterdir()):
    print(" ", p)

print("\nDONE.")

