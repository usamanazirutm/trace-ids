#!/bin/bash
# Reproduces every run reported in the paper. Each command is resumable: re-run it after an interruption.
set -e
cd "$(dirname "$0")/.."
UN="--dataset unsw --path data/unsw"
CI="--dataset cic --path data/cic/ciciot2023_clean.parquet --cap 10000"
CS="--n_tp 1000 --n_bench 600 --n_sens 200"
python -m trace_ids.pipeline $UN --detector cnnbilstm --seed 0 --full --out results/unsw_cnnbilstm_s0
python -m trace_ids.pipeline $UN --detector mlp       --seed 0 --full --out results/unsw_mlp_s0
python -m trace_ids.pipeline $CI --detector cnnbilstm --seed 0 --full $CS --out results/cic_cnnbilstm_s0
python -m trace_ids.pipeline $CI --detector mlp       --seed 0 --full $CS --out results/cic_mlp_s0
for s in 1 2; do
  python -m trace_ids.pipeline $UN --detector mlp --seed $s --out results/unsw_mlp_s$s
  python -m trace_ids.pipeline $CI --detector mlp --seed $s $CS --out results/cic_mlp_s$s
done
