#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2()'
