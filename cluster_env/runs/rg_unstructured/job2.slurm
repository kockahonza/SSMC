#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#requesting one task of ? cpus (on one node implied)
#SBATCH --cpus-per-task=101

time julia -t $SLURM_CPUS_PER_TASK --project -L job.jl -e 'main_run4_N5_lDs_positive()'
# time julia -t 20 --project -L job.jl -e 'ltest_run1()' # for testing
