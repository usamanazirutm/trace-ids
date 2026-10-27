# TRACE — Traffic-Realisable Counterfactual Explanations for IoT Intrusion Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23681520.svg)](https://doi.org/10.5281/zenodo.23681520)
[![Release](https://img.shields.io/badge/release-tracev1-blue)](https://github.com/usamanazirutm/trace-ids/releases/tag/tracev1)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Code accompanying the manuscript

> **Nazir, M. U., & Ngadi, M. A.** *TRACE: Traffic-Realisable Counterfactual Explanations for Deep Learning-Based IoT Intrusion Detection* (under review).

TRACE explains an intrusion-detection alert by finding the closest **benign flow that could actually exist**. A counterfactual is only returned if it respects the semantics of the flow features:

| Constraint family | Example (UNSW-NB15 / CICIoT2023) |
|---|---|
| C1 immutability | protocol, service, connection state, protocol-indicator fractions, window size never change |
| C2 integrality | packet, byte and flag counts stay integers |
| C3 bounds | every feature stays inside its training range (e.g. TTL ≤ 255) |
| C4 relations | `sloss ≤ spkts`, `Min ≤ Max`, `flag counts ≤ Number`, bytes within packet-size limits |
| C5 derived features | `smean = sbytes/spkts`, `rate = (spkts+dpkts−1)/dur`, `tcprtt = synack + ackdat`, `AVG = Tot sum/Number`, `Variance = Std²` |

TRACE searches twice (from the alert and from the nearest matched benign flow) and keeps the better feasible result. Every candidate relation is **verified on the training split** (kept only if it holds on ≥ 99.5 % of flows) before it is enforced (`trace_ids/space.py`). A differentiable **repair map** projects every search iterate onto the feasible set, a **local plausibility term** pulls the counterfactual towards benign flows with the same immutable attributes, and a **greedy reversion** step makes it sparse (`trace_ids/cf.py`).

Baselines included: Wachter-style gradient counterfactuals, Growing Spheres, nearest benign neighbour (NUN), and three TRACE ablations. Detectors: a CNN-BiLSTM and an MLP (PyTorch); Random Forest and XGBoost references.

---

## 1. Installation

```bash
git clone https://github.com/usamanazirutm/trace-ids.git
cd trace-ids

# option A: conda (CPU)
conda env create -f environment.yml
conda activate trace-ids

# option B: pip (Python 3.10–3.12)
pip install -r requirements.txt
```

A GPU is not needed. All reported experiments ran on **one CPU core with 3 GB RAM**.

## 2. Data

See [`data/README.md`](data/README.md). Expected layout:

```
data/unsw/UNSW_NB15_training-set.csv
data/unsw/UNSW_NB15_testing-set.csv
data/cic/ciciot2023_clean.parquet        # or a folder of MERGED_CSV / part-*.csv files
```

## 3. Run one experiment

```bash
python -m trace_ids.pipeline --dataset unsw --path data/unsw \
       --detector cnnbilstm --seed 0 --full --out results/unsw_cnnbilstm_s0

python -m trace_ids.pipeline --dataset cic --path data/cic/ciciot2023_clean.parquet --cap 10000 \
       --detector cnnbilstm --seed 0 --full --n_tp 1000 --n_bench 600 --n_sens 200 \
       --out results/cic_cnnbilstm_s0
```

The pipeline is **resumable**: every training epoch and every counterfactual batch is saved, so an interrupted run continues where it stopped when the same command is repeated. It prints `PIPELINE COMPLETE` when finished. `--budget SECONDS` limits the run time of one invocation.

| Flag | Meaning |
|---|---|
| `--detector {cnnbilstm,mlp}` | differentiable detector to explain |
| `--full` | adds Growing Spheres, TRACE ablations, β/γ sensitivity grid, RF/XGBoost references |
| `--cap N` | cap every attack class at N flows (all benign flows kept) |
| `--n_tp N` | true alerts sampled per split (all false alarms are always explained) |
| `--steps T --beta β --gamma γ` | TRACE search budget and weights (defaults 200, 0.05, 0.05) |
| `--det_only` | stop after detection metrics |

## 4. Reproduce the paper

```bash
bash scripts/run_all.sh                         # all runs used in the paper
python -m trace_ids.report --results results --out paper_assets
```

`trace_ids.report` writes the LaTeX table bodies, a macro file with every number quoted in the manuscript, and the figures.

## 5. Outputs (per run directory)

| File | Content |
|---|---|
| `data_stats.json` | raw / deduplicated / conflict-free / used counts, split sizes |
| `constraints.csv` | every candidate relation with train and test satisfaction and whether it is enforced |
| `features.csv` | per-feature flags (immutable, derived, integer, log-scaled), ranges and MAD scales |
| `detection.json`, `per_class*.csv`, `reference_models.csv` | detection metrics |
| `cf_quality.csv` | validity, feasibility (all and per family), proximity, sparsity, plausibility, runtime |
| `sensitivity.csv` | β × γ grid |
| `change_profile_*.csv` | mean normalised change per feature and alert group |
| `case_studies.csv` | median-cost false-alarm and true-alert explanations in raw units |
| `ig_agreement.json` | agreement between counterfactual changes and Integrated Gradients |

## 6. Repository structure

```
trace_ids/
  data.py       loading, deduplication, label-conflict removal, class capping, stratified split
  space.py      preprocessing, verify-then-enforce constraints, differentiable repair map, feasibility checks
  models.py     CNN-BiLSTM and MLP detectors, training loop
  cf.py         TRACE, Wachter-style, Growing Spheres, NUN, k-NN plausibility, Integrated Gradients
  metrics.py    detection and counterfactual-quality metrics
  pipeline.py   resumable end-to-end experiment
  report.py     tables, macros and figures for the manuscript
scripts/        run_all.sh, make_report.sh
data/           dataset instructions (data files are not tracked)
results/        summary outputs of the runs reported in the paper
docs/           REPRODUCIBILITY.md
```

## 7. Responsible use

A counterfactual explanation states which changes would make a flagged flow look benign to the detector. The same information can guide evasion. Show TRACE explanations only to authorised analysts, never to the traffic source, and do not expose them through public interfaces.

## 8. Citation

If you use this code, please cite the paper (details will be updated on publication) and the software archive:

> Nazir, M. U., & Ngadi, M. A. (2026). *TRACE: Traffic-Realisable Actionable Counterfactual Explanations for IoT Intrusion Detection* (release tracev1). Zenodo. https://doi.org/10.5281/zenodo.23681520

```bibtex
@software{nazir2026trace,
  author    = {Nazir, Muhammad Usama and Ngadi, Md Asri},
  title     = {TRACE: Traffic-Realisable Actionable Counterfactual Explanations for IoT Intrusion Detection},
  year      = {2026},
  version   = {tracev1},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23681520},
  url       = {https://doi.org/10.5281/zenodo.23681520}
}
```

## 9. License

MIT — see [`LICENSE`](LICENSE). The datasets are subject to their own terms (see `data/README.md`).
