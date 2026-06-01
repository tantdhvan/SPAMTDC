#!/usr/bin/env bash

#SBATCH --job-name=lt
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

set -euo pipefail

BIN=${BIN:-./lt}
GRAPH=${GRAPH:-data/fb_suit.bin}
CSV=${CSV:-lt_fb_suit_active.csv}
B_FACTOR=${1:-0.2}
THREADS=${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-16}}
LT_MC=${LT_MC:-100}
LT_EVAL_MC=${LT_EVAL_MC:-10000}
SEED=${SEED:-42}
PROFIT_MU=${PROFIT_MU:-1}
SUIT_SEED=${SUIT_SEED:-20260528}
ETA_MIN=${ETA_MIN:-0.2}
ETA_MAX=${ETA_MAX:-0.8}
XI_MAX=${XI_MAX:-0.2}
ROUTES=${ROUTES:-"ops gong pham"}
STATE_DIR=${STATE_DIR:-state}
CSV_BASE=$(basename "$CSV" .csv)
SUIT_ARGS=(--profit-mu "$PROFIT_MU" --suit-seed "$SUIT_SEED" --eta-min "$ETA_MIN" --eta-max "$ETA_MAX" --xi-max "$XI_MAX")

mkdir -p "$STATE_DIR"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=${OMP_PROC_BIND:-spread}
export OMP_PLACES=${OMP_PLACES:-cores}

for ROUTE in $ROUTES; do
  case "$ROUTE" in
    ops)
      "$BIN" "$GRAPH" ops "$B_FACTOR" --alpha 1.505 "${SUIT_ARGS[@]}" --threads "$THREADS" --seed "$SEED" --lt-mc "$LT_MC" --eval-mc "$LT_EVAL_MC" --csv "$CSV"
      ;;
    gong)
      "$BIN" "$GRAPH" gong "$B_FACTOR" --alpha 0.4 --epsilon 0.1 "${SUIT_ARGS[@]}" --threads "$THREADS" --seed "$SEED" --lt-mc "$LT_MC" --eval-mc "$LT_EVAL_MC" --csv "$CSV" --state "$STATE_DIR/${CSV_BASE}.gong.state"
      ;;
    pham)
      "$BIN" "$GRAPH" pham "$B_FACTOR" --alpha 0.4 --epsilon 0.1 "${SUIT_ARGS[@]}" --threads "$THREADS" --seed "$SEED" --lt-mc "$LT_MC" --eval-mc "$LT_EVAL_MC" --csv "$CSV" --state "$STATE_DIR/${CSV_BASE}.pham.state"
      ;;
    *)
      echo "Unknown route in ROUTES: $ROUTE" >&2
      exit 1
      ;;
  esac
done
