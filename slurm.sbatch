#!/bin/bash

#SBATCH --job-name=picc-rl-array
#SBATCH --output=slurm_logs/picc-rl_%A_%a.log
#SBATCH --error=slurm_logs/picc-rl_%A_%a.log
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --array=1-2

echo "Job Array ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"
echo "Running on node: $(hostname)"
echo "Script location: ${SCRIPT_DIR}"
echo "Started at: $(date)"

mkdir -p "${SCRIPT_DIR}/slurm_logs"

module load miniforge/24.11.2-py312

source activate tool_transfer_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

CONFIG_FILE="exp_configs/${SLURM_ARRAY_TASK_ID}.json"
echo "Using configuration file: ${CONFIG_FILE}"

python3 main.py --config "${CONFIG_FILE}"

echo "Finished at: $(date)"  # $ tail -f logs/picc-rl_15662689_1.log
# sbatch slurm.sbatch
# watch squeue -u $(whoami)  (or without watch)  # check job conditions

