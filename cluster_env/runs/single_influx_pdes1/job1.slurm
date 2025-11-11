#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=128

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2()'
