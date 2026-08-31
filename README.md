# Synthetic Image Oversampling for Rare-Class Representation Learning in Deep Facial Phenotyping

Code and reproducibility materials for evaluating synthetic facial-image
oversampling in long-tailed rare-disease classification.

The study evaluates five training conditions (Baseline, PD25, PD50, PD75,
and PD100) using two complementary facial phenotyping architectures:

- **GestaltMatcher-Arc** — image-based classification using a pretrained
  Glint360K R50 face representation network.
- **GestaltMML** — multimodal classification using facial images and
  phenotype-related clinical text.

All conditions use the same fixed patient-disjoint real evaluation partition.
Synthetic images are used only for training augmentation.

## Repository structure

    arc/       GestaltMatcher-Arc training, evaluation, dataset, and embedding analysis
    mml/       GestaltMML training and evaluation
    data/      Data setup and availability documentation
    results/   Lightweight numerical results reported in the study

## Dataset

The experiments use a 10-disease subset of the GestaltMatcher Database:

- 1,289 real training images
- 558 real evaluation images
- 10 disease classes
- zero patient and image overlap between training and evaluation

Synthetic augmentation uses PDIDB images at increasing fractions of the
available synthetic pool for each disease:

| Condition | Synthetic images | Total training images |
|-----------|-----------------:|----------------------:|
| Baseline  | 0                | 1,289 |
| PD25      | 433              | 1,722 |
| PD50      | 866              | 2,155 |
| PD75      | 1,298            | 2,587 |
| PD100     | 1,727            | 3,016 |

See `data/README.md` for data configuration and availability.

## Main results

### GestaltMatcher-Arc

The highest observed overall Top-1 accuracy occurred at PD50:

- Baseline: **88.17%**
- PD50: **90.86%**
- Change: **+2.69 percentage points**

The cosine silhouette score of the 512-dimensional representations increased
from **0.2283** at Baseline to **0.2669** at PD50. Nearest-centroid separation
increased for 9 of 10 disease classes.

### GestaltMML

Performance continued improving through the largest tested augmentation level:

- Baseline: **67.20%**
- PD100: **75.27%**
- Change: **+8.07 percentage points**

Complete classification and embedding results are provided in `results/`.

## Data configuration

Patient facial images and restricted clinical data are not included.

Set local data locations before running the experiments:

    export GMDB_IMAGE_ROOT=/path/to/gmdb_crops
    export GMDB_DATASET_ROOT=/path/to/GestaltMatcherDB

See `data/README.md` for details.

## GestaltMatcher-Arc

Run Arc commands from the `arc/` directory:

    cd arc

Build the fixed 10-disease dataset:

    python build_gmdb10_v2.py

Build the synthetic augmentation conditions:

    python build_gmdb10_pd_v2.py

Training is performed with:

    python train_gm_arc.py [arguments]

Final disease-specific evaluation:

    python evaluate_arc_v2_per_disease.py

Baseline-versus-PD50 embedding analysis:

    python analyze_arc_embeddings_v2.py

## GestaltMML

GestaltMML training is implemented in:

    mml/train_gestaltmml_10d.py

SLURM scripts reproduce the five experimental conditions:

    mml/run_gestaltmml_v2_baseline.slurm
    mml/run_gestaltmml_v2_pd25.slurm
    mml/run_gestaltmml_v2_pd50.slurm
    mml/run_gestaltmml_v2_pd75.slurm
    mml/run_gestaltmml_v2_pd100.slurm

Disease-specific evaluation is implemented in:

    mml/evaluate_mml_v2_per_disease.py

## Reproducibility

The principal experiments use random seed 11. All augmentation conditions
share the same fixed real training/evaluation partition, and synthetic images
are added only to the training partition.

Exact numerical summaries used in the study are available under `results/`.

## Upstream software

Parts of the GestaltMatcher-Arc implementation in `arc/lib/` are derived from
the GestaltMatcher-Arc project and are redistributed under its applicable
Creative Commons Attribution-NonCommercial 4.0 International license.

Additional third-party datasets, pretrained models, and software remain
subject to their respective licenses and usage conditions.

## License

Licensing and attribution information for redistributed upstream components
is provided with this repository.
