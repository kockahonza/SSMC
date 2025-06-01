#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#requesting one task of ? cpus (on one node implied)
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G

time julia -t $SLURM_CPUS_PER_TASK --project -L job.jl -e 'add_animations_job()'
