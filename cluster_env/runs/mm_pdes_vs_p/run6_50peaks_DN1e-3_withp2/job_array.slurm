#!/bin/bash
#SBATCH -J mm_run6
#SBATCH -p BIOP
#SBATCH -o logs/%A_%a.out
#SBATCH --time 7:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G   # 2G aborts julia's JIT ("protect_page: Invalid argument") on the 40-core nodes
#SBATCH --array=1-60

# dias (Slurm). Twin of job_array.qsub: one row per task, one core per task.
# Note --mem here is per task, not per node as it is in a non-array job.
#
# See job_array.qsub for why this is an array of single-core tasks rather than
# one parallel job.

mkdir -p logs
time julia -t 1 --project -L base.jl -e "main1_row($SLURM_ARRAY_TASK_ID)"
