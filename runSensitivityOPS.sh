#!/usr/bin/env bash

#SBATCH --job-name=SensOPS
#SBATCH --partition=normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

BIN=${BIN:-./kic}
GRAPH=${GRAPH:-data/email_suit.bin}
B_FACTOR=${1:-0.2}
CSV=${2:-sensitivity_suit_ops.csv}
SEED=${SEED:-42}
KIC_MC=${KIC_MC:-200}
KIC_EVAL_MC=${KIC_EVAL_MC:-10000}
THREADS=${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-16}}
PROFIT_MU=${PROFIT_MU:-1}
SUIT_SEED=${SUIT_SEED:-20260528}
ETA_MIN=${ETA_MIN:-0.2}
ETA_MAX=${ETA_MAX:-0.8}
XI_MAX=${XI_MAX:-0.2}
SUIT_ARGS=(--profit-mu "$PROFIT_MU" --suit-seed "$SUIT_SEED" --eta-min "$ETA_MIN" --eta-max "$ETA_MAX" --xi-max "$XI_MAX")

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}

DS=(1.1 1.25 1.505 1.75 2.0 2.5)

for D in "${DS[@]}"; do
  "$BIN" "$GRAPH" ops "$B_FACTOR" --alpha "$D" "${SUIT_ARGS[@]}" --threads "$THREADS" --seed "$SEED" --kic-mc "$KIC_MC" --eval-mc "$KIC_EVAL_MC" --csv "$CSV"
done
