#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 10:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main4()'
