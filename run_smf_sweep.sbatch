#!/bin/bash -l
#SBATCH --job-name=smf_p0_sweep
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=logs/%x-%A_%a.out

set -euo pipefail

# --- Modules & env ---
[ -f /etc/profile.d/modules.sh ] && source /etc/profile.d/modules.sh
module load applications/conda/miniconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate smf_env

# --- Inputs from submit script ---
: "${P0_START:?Need P0_START}"
: "${P0_END:?Need P0_END}"
: "${N_INTERVALS:?Need N_INTERVALS}"
INDEX_OFFSET="${INDEX_OFFSET:-0}"
SIM_NAME="${SIM_NAME:-default_2d_run}"

if ! [[ "${N_INTERVALS}" =~ ^[0-9]+$ ]]; then
  echo "N_INTERVALS must be a non-negative integer, got '${N_INTERVALS}'" >&2
  exit 1
fi

NUM_PUMP_FRAMES=$(( N_INTERVALS + 1 ))

TID="${SLURM_ARRAY_TASK_ID:?missing array id}"
INDEX=$(( INDEX_OFFSET + TID ))

# --- Log what we're about to run ---
echo "JOB $SLURM_JOB_ID ARRAY_ID $TID  -> i=${INDEX} ; range=[${P0_START}, ${P0_END}] ; intervals=${N_INTERVALS} (frames=${NUM_PUMP_FRAMES})"
echo "Simulation name: ${SIM_NAME}"
echo "Analytic t0 extension enabled (--extend-time-using-t0)."

# --- Run job (your script computes p0 from s,e,n,i) ---
cd /home/users/seb25178/Projects/BEC_SMF_2D/patt2d_smf_BEC_sim
mkdir -p logs

python patt2d_q_sfm.py \
  -f "${SIM_NAME}" \
  -s "${P0_START}" \
  -e "${P0_END}" \
  -n "${NUM_PUMP_FRAMES}" \
  -i "${INDEX}" \
  --extend-time-using-t0
