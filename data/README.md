# Data

This repository does not redistribute patient facial images, clinical metadata,
or model checkpoints.

## Dataset

Experiments use a fixed patient-disjoint 10-disease subset of the
GestaltMatcher Database (GMDB):

- 1,289 real training images
- 558 real evaluation images
- 10 disease classes
- no patient or image overlap between training and evaluation

Synthetic PDIDB images are used only for training augmentation. The real
evaluation partition remains fixed across all conditions.

## Augmentation conditions

- Baseline: 0 synthetic images
- PD25: +433 synthetic images
- PD50: +866 synthetic images
- PD75: +1,298 synthetic images
- PD100: +1,727 synthetic images

PD25 through PD100 represent increasing fractions of each disease's available
synthetic-image pool.

## Local configuration

Set the location of aligned real GMDB images with:

    export GMDB_IMAGE_ROOT=/path/to/gmdb_crops

For GestaltMML, set the dataset directory with:

    export GMDB_DATASET_ROOT=/path/to/GestaltMatcherDB

The canonical split and augmented datasets can be constructed from the
scripts in `arc/`.

## Data availability

Users must obtain the underlying real and synthetic datasets under their
respective access and usage conditions. Patient facial imagery and restricted
clinical data are intentionally not included in this repository.
