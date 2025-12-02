#!/usr/bin/env bash
# Usage: ./submit_smf_sweep.sh -s P0_START -e P0_END -n N_INTERVALS [-i INDEX_OFFSET] [--sim-name name] [--extend-sim] [--restart-index idx]
set -euo pipefail

usage() {
  cat >&2 <<USAGE
Usage: $0 -s P0_START -e P0_END -n N_INTERVALS [-i INDEX_OFFSET] [--sim-name name] [--extend-sim] [--restart-index idx]

  -s    Start pump saturation parameter (required)
  -e    End pump saturation parameter (required)
  -n    Number of intervals between start and end (required)
  -i    Index offset for array jobs (default: 0)
  --sim-name   Simulation name (default: default_2d_run)
  --extend-sim Enable analytic time extension (--extend-time-using-t0)
  --restart-index  Use an existing psi_phase/psi_not_squared{idx}_*.out as initial state (--restart-from-index idx)
USAGE
  exit 1
}

P0_START=""
P0_END=""
N_INTERVALS=""
INDEX_OFFSET="0"
SIM_NAME="default_2d_run"
EXTEND_SIM="false"
RESTART_INDEX=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    -s)
      P0_START="${2:-}" || true
      shift 2
      ;;
    -e)
      P0_END="${2:-}" || true
      shift 2
      ;;
    -n)
      N_INTERVALS="${2:-}" || true
      shift 2
      ;;
    -i)
      INDEX_OFFSET="${2:-}" || true
      shift 2
      ;;
    --sim-name)
      SIM_NAME="${2:-}" || true
      shift 2
      ;;
    --extend-sim)
      EXTEND_SIM="true"
      shift 1
      ;;
    --restart-index)
      RESTART_INDEX="${2:-}" || true
      shift 2
      ;;
    *)
      echo "Unrecognised argument: $1" >&2
      usage
      ;;
  esac
done

# Required arguments
[ -n "$P0_START" ] && [ -n "$P0_END" ] && [ -n "$N_INTERVALS" ] || usage

if ! [[ "$N_INTERVALS" =~ ^[0-9]+$ ]]; then
  echo "Error: N_INTERVALS must be a non-negative integer, got '$N_INTERVALS'" >&2
  exit 1
fi

if ! [[ "$INDEX_OFFSET" =~ ^-?[0-9]+$ ]]; then
  echo "Error: INDEX_OFFSET must be an integer, got '$INDEX_OFFSET'" >&2
  exit 1
fi

if [ -n "$RESTART_INDEX" ] && ! [[ "$RESTART_INDEX" =~ ^-?[0-9]+$ ]]; then
  echo "Error: RESTART_INDEX must be an integer when provided, got '$RESTART_INDEX'" >&2
  exit 1
fi

export P0_START P0_END N_INTERVALS
export SIM_NAME INDEX_OFFSET EXTEND_SIM RESTART_INDEX

cd /home/users/seb25178/Projects/BEC_SMF_2D/patt2d_smf_BEC_sim
mkdir -p logs

# If N_INTERVALS = M (number of intervals), indices should be 0..M (inclusive) => M+1 total values.
ARRAY_SPEC="0-${N_INTERVALS}"

sbatch \
  --chdir=/home/users/seb25178/Projects/BEC_SMF_2D/patt2d_smf_BEC_sim \
  --export=ALL \
  --array="${ARRAY_SPEC}" \
  run_smf_sweep.sbatch
