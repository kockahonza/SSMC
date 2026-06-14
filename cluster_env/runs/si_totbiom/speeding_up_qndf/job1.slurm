#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 120:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
