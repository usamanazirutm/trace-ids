# Reproducibility notes

## Environment used for the reported numbers
- 1 CPU core, 3 GB RAM, no GPU (Ubuntu 24.04)
- Python 3.12, PyTorch 2.14.1 (CPU build, conda-forge), scikit-learn 1.9.1, XGBoost 2.1.4,
  NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, matplotlib 3.11.2, pyarrow 25.0.0
- `torch.set_num_threads(1)` (override with the environment variable `TRACE_THREADS`)

## Determinism
- One stratified 70/15/15 split with seed 0, shared by every detector.
- Detector seeds (`--seed`) change only weight initialisation and mini-batch order.
- Alert subsampling, the benchmark subset and the sensitivity subset are drawn with `RandomState(seed)`
  and stored in `alerts.npz`, so every method is compared on identical alerts.
- Interrupted runs resume from `train_state.pt` and the per-batch `cf_*.npz` files.

## Approximate wall-clock time on 1 CPU core
| Step | UNSW-NB15 | CICIoT2023 (capped) |
|---|---|---|
| CNN-BiLSTM training | ~17 min (22 epochs) | ~15 min |
| TRACE, CNN-BiLSTM, per 1,000 alerts | ~1.7 min | ~1.7 min |
| Wachter-style, per 1,000 alerts | ~0.9 min | ~0.9 min |
| MLP run incl. counterfactuals | ~5 min | ~4 min |

## What each run writes
See the table in the main README. Large intermediate files (`*.npz`, `*.pt`, `*.pkl`) are excluded by
`.gitignore`; the summary files in `results/` are those reported in the paper.

## Rebuilding the manuscript numbers
`bash scripts/make_report.sh` regenerates `paper_assets/generated_macros.tex` (every number quoted in
the text), `paper_assets/generated_tables.json` (table bodies) and the figures.
