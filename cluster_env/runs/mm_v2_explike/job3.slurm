#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem=150G

time julia -t $SLURM_CPUS_PER_TASK --project -L v2.jl -e 'v2main_highN0_extra()'
