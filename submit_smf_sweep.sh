#!/usr/bin/env bash
# Usage: ./submit_smf_sweep.sh P0_START P0_END N_INTERVALS [--sim-name name]
set -euo pipefail
[ "$#" -ge 3 ] || { echo "Usage: $0 P0_START P0_END N_INTERVALS [--sim-name name]" >&2; exit 1; }

P0_START="$1"
P0_END="$2"
N_INTERVALS="$3"
shift 3

SIM_NAME="default_2d_run"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --sim-name)
      shift
      [ "$#" -gt 0 ] || { echo "Error: --sim-name requires a value." >&2; exit 1; }
      SIM_NAME="$1"
      ;;
    *)
      echo "Unrecognised argument: $1" >&2
      echo "Usage: $0 P0_START P0_END N_INTERVALS [--sim-name name]" >&2
      exit 1
      ;;
  esac
  shift
done

export P0_START P0_END N_INTERVALS
export SIM_NAME

cd /home/users/seb25178/Projects/BEC_SMF_2D/patt2d_smf_BEC_sim
mkdir -p logs

# If N_INTERVALS = M (number of intervals), indices should be 0..M (inclusive) => M+1 total values.
ARRAY_SPEC="0-${N_INTERVALS}"

sbatch \
  --chdir=/home/users/seb25178/Projects/BEC_SMF_2D/patt2d_smf_BEC_sim \
  --export=ALL \
  --array="${ARRAY_SPEC}" \
  run_smf_sweep.sbatch
