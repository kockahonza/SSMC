#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 2:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G

# Same node shape as job2/job3 so the contention is the real one: 10 rows on 40
# cores in 60G. Three 5 minute waves plus Julia startup and compile, well under
# the 2h.

time julia -t $SLURM_CPUS_PER_TASK --project -L bench_solver_threads.jl -e 'bench_solver_threads()'
