#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 10:00:00
#SBATCH --cpus-per-task=26
#SBATCH --mem=200G

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1(); fit_main1()'
