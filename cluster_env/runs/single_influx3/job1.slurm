#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=92G
#SBATCH -w compute-0-9

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
