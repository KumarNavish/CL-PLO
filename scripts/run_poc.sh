#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

printf "Running validation experiment...\n"
python3 validation_experiment.py

printf "\nGenerated artifacts:\n"
printf " - figs/validation_wide.png\n"
printf " - figs/validation_wide.pdf\n"
printf " - figs/metrics.tex\n"
