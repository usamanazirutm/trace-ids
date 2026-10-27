# Datasets

Data files are **not** tracked in this repository. Download them and place them as shown.

## UNSW-NB15

Source: UNSW Canberra Cyber Range Lab, https://research.unsw.edu.au/projects/unsw-nb15-dataset
(Moustafa & Slay, MilCIS 2015). Free use for academic research.

Use the two partitioned files:

    data/unsw/UNSW_NB15_training-set.csv
    data/unsw/UNSW_NB15_testing-set.csv

Both files are merged (257,673 flows). `attack_cat` is the 10-class label; `id`, `label`, `stcpb` and
`dtcpb` are dropped. Exact duplicates and feature vectors with conflicting labels are removed
automatically (257,673 -> 162,744 -> 151,238 flows).

## CICIoT2023

Source: Canadian Institute for Cybersecurity, https://www.unb.ca/cic/datasets/iotdataset-2023.html
(Neto et al., Sensors 23(13):5941, 2023). Subject to the CIC terms of use.

The paper uses a public cleaned copy of the official MERGED_CSV files (Merged01 + Merged02; NaN/inf rows,
exact duplicates, multiply labelled vectors and two redundant columns removed by its curator;
918,764 flows, 37 features, 34 classes):

    https://github.com/arpangl/IoT2026/raw/main/lab2/data/ciciot2023_clean.parquet
    -> data/cic/ciciot2023_clean.parquet

`--cap 10000` keeps every benign flow and caps each attack class at 10,000 flows (231,384 flows).
The loader also accepts a folder of official CSV files (`--path data/cic/<folder>`). Constraints are
re-verified on whatever release is supplied; relations that do not hold are not enforced.
