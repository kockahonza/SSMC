#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#requesting one task of ? cpus (on one node implied)
#SBATCH --cpus-per-task=128

time julia -t $SLURM_CPUS_PER_TASK --project -L job.jl -e 'main_N10_comprehensive1()'
