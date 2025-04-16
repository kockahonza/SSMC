#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#requesting one task of ? cpus (on one node implied)
#SBATCH --cpus-per-task=128

# time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'job1()'
time julia -t 20 --project -L base.jl -e 'job1()'
