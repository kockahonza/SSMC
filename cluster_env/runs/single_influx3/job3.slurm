#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-10

# main3_NM10: same 20 x 10 x 120 grid as job2 but at N=M=10 instead of the
# default 20, so the state vectors are half the length and this is the lighter
# of the two. Same 60G sizing on the 64039M nodes for the same reason.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main3_NM10()'
