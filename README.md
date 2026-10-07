# Synthetic Image Oversampling for Rare-Class Representation Learning in Deep Facial Phenotyping

Code and reproducibility materials for evaluating synthetic facial-image
oversampling in long-tailed rare-disease classification.

This study evaluates five training conditions (Baseline, PD25, PD50, PD75,
and PD100) using two complementary facial phenotyping architectures:

- **GestaltMatcher-Arc** — image-based classification using a pretrained
  Glint360K R50 facial representation network.
- **GestaltMML** — multimodal classification using facial images and
  phenotype-related clinical text.

All conditions use the same fixed patient-disjoint real evaluation partition.
Synthetic images are used only for training augmentation.

## Repository structure

    arc/       GestaltMatcher-Arc training, evaluation, embedding, and audit code
    mml/       GestaltMML training and evaluation
    data/      Data documentation and public PDIDB augmentation manifests
    results/   Numerical results and sensitivity/audit summaries

## Dataset

Experiments use a 10-disease subset of the GestaltMatcher Database:

- 1,289 real training images
- 558 real evaluation images
- 10 disease classes
- patient-disjoint training and evaluation partitions

Synthetic augmentation uses PDIDB images at increasing fractions of the
available synthetic pool:

| Condition | Synthetic images | Total training images |
|-----------|-----------------:|----------------------:|
| Baseline  | 0     | 1,289 |
| PD25      | 433   | 1,722 |
| PD50      | 866   | 2,155 |
| PD75      | 1,298 | 2,587 |
| PD100     | 1,727 | 3,016 |

The augmentation sets are nested:

`PD25 ⊂ PD50 ⊂ PD75 ⊂ PD100`

Exact public PDIDB identifiers for each condition are provided under
`data/pdidb_manifests/`.

Patient facial images and restricted clinical data are not redistributed.

## Main results

All classification experiments were repeated using training seeds **11, 22,
and 33**. The real-data split and synthetic subsets were held fixed across
training seeds. Values below are mean ± sample SD.

### GestaltMatcher-Arc

Overall Top-1 accuracy:

- Baseline: **89.19 ± 0.92%**
- PD25: **89.96 ± 0.65%**
- PD50: **90.20 ± 0.58%**
- PD75: **90.02 ± 0.45%**
- PD100: **89.84 ± 0.52%**

The highest observed mean overall Top-1 accuracy occurred at PD50.

The representative seed-11 Baseline-versus-PD50 embedding analysis showed an
increase in cosine silhouette score from **0.2283** to **0.2669**. Quantitative
embedding metrics were computed in the original 512-dimensional representation
space.

### GestaltMML

Overall Top-1 accuracy increased across the tested augmentation levels:

- Baseline: **68.82 ± 1.53%**
- PD25: **69.65 ± 1.45%**
- PD50: **72.16 ± 1.77%**
- PD75: **73.66 ± 1.35%**
- PD100: **74.31 ± 0.90%**

Synthetic images in GestaltMML were paired only with placeholder text rather
than additional clinical-text information.

Complete aggregate results are provided under `results/`.

## Class-weighting sensitivity analysis

To test whether the GestaltMatcher-Arc augmentation effect depended on
class-frequency-weighted cross-entropy, Baseline and PD50 were additionally
evaluated with unweighted cross-entropy across the same three training seeds.

- Weighted CE paired PD50 gain: **+1.02 ± 1.49 percentage points**
- Unweighted CE paired PD50 gain: **+1.02 ± 1.72 percentage points**

The similar gains indicate that the modest Arc augmentation effect was not
primarily attributable to class-frequency weighting.

## Leakage audit

The full 1,727-image PDIDB synthetic pool was screened against the 558-image
real evaluation partition using:

1. exact pixel-level hashes;
2. perceptual hashes;
3. nearest-neighbor cosine similarity in the 512-dimensional
   GestaltMatcher-Arc representation space.

No exact pixel duplicates were identified.

Synthetic-to-evaluation nearest-neighbor similarities were generally lower
than the real-training-to-evaluation patient-disjoint reference distribution:

- synthetic → evaluation median: **0.591**
- real training → evaluation median: **0.670**
- synthetic → evaluation maximum: **0.851**
- real training → evaluation maximum: **0.898**

Manual inspection of the closest perceptual and embedding matches found no
apparent duplicate images. Restricted patient-face contact sheets are not
redistributed.

Numerical audit summaries are available under `results/`.

## GestaltMatcher-Arc

Run Arc commands from:

    cd arc

Training:

    python train_gm_arc.py [arguments]

Canonical multi-seed evaluation:

    python evaluate_arc_multiseed.py

Class-weighting sensitivity analysis:

    python evaluate_arc_unweighted_ablation.py

Representative Baseline-versus-PD50 embedding analysis:

    python analyze_arc_embeddings_v2.py

Leakage screening:

    python leakage_check.py

## GestaltMML

GestaltMML training is implemented in:

    mml/train_gestaltmml_10d.py

Training configurations for the five augmentation conditions are provided in
the corresponding SLURM scripts under `mml/`.

Disease-specific evaluation is implemented in:

    mml/evaluate_mml_v2_per_disease.py

## Reproducibility

Classification experiments use training seeds **11, 22, and 33**.

The following are fixed across training seeds:

- patient-disjoint GMDB training/evaluation partition;
- PDIDB synthetic subsets;
- model configuration within each architecture.

Synthetic subset selection uses sampling seed **11**.

For each training run, the selected checkpoint is the epoch with the highest
**overall Top-1 accuracy** on the fixed evaluation partition. If multiple
epochs have identical overall Top-1 accuracy, the earliest epoch is selected.

The same evaluation partition is used for checkpoint selection and reported
classification metrics, as described in the manuscript.

Exact synthetic-image manifests are provided under:

    data/pdidb_manifests/

The manifests expose public PDIDB image identifiers so the nested augmentation
sets can be reconstructed without access to restricted GMDB data.

Restricted GMDB patient images, clinical data, and trained checkpoints are not
redistributed.

## Data configuration

Set local data locations before running experiments:

    export GMDB_IMAGE_ROOT=/path/to/gmdb_crops
    export GMDB_DATASET_ROOT=/path/to/GestaltMatcherDB

See `data/README.md` for additional information.

## Upstream software

Parts of the GestaltMatcher-Arc implementation in `arc/lib/` are derived from
the GestaltMatcher-Arc project and are redistributed under its applicable
Creative Commons Attribution-NonCommercial 4.0 International license.

Additional third-party datasets, pretrained models, and software remain
subject to their respective licenses and usage conditions.

## License

Licensing and attribution information for redistributed upstream components is
provided with this repository.
