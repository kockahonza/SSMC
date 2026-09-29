#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-11

# main7_B20: same 20 x 10 x 120 grid as job2/job3/job4 but at B=20 instead of
# main2's B=5. Same 60G sizing on the 64039M nodes for the same reason.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main7_B20()'
