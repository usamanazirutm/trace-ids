#!/bin/bash
# Writes LaTeX table bodies, number macros and figures for the manuscript.
cd "$(dirname "$0")/.."
python -m trace_ids.report --results results --out paper_assets
