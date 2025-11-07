#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=128largem_lowDR4
#SBATCH --mem=150G

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'v2main_highN0_largem_lowDR()'
