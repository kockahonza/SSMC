#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G
#SBATCH -w compute-0-7

# A benchmark twin of job1: main4_part1_st4 runs the exact same 20 rows as
# main4_part1 (systems 1,4,...,28) with the same tolerances and maxtime, changing
# only solver_threads=1 -> 4, and writes to `data_st4` so neither job can see or
# skip the other's output. Comparing realtime row for row between `data` and
# `data_st4` is the whole point.
#
# So this one does want the full node: 10 rows x 4 solver threads = 40 cpus, and
# --mem=94G of compute-0-7's 96G because the earlier sN=2500 benchmark OOMed
# inside umfpack_numeric! with solver_threads=4 at 10 rows in 60G. The fairshare
# argument for trimming cpus does not apply here - these 40 are actually used.
#
# Only worth running while a 96G node is idle; it duplicates work job1 is already
# doing, so if the partition fills up this is the job to cancel.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main4_part1_st4()'
