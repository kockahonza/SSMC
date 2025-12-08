#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=300G

time julia -t $SLURM_CPUS_PER_TASK --project -L v4.jl -e 'v4main_base()'
