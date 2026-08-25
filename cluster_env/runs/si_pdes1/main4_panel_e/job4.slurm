#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 48:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G
#SBATCH -w compute-0-6

# main4_all_st4 runs all 60 rows with run_threads=10, solver_threads=4, into
# data_st4_all - separate from the three production parts writing `data`, so
# neither can skip the other's rows.
#
# The whole of a 96G node: 10 rows x 4 solver threads = its 40 cores, and 94G is
# 96256M of the node's 96293M, ie everything schedulable. The fairshare argument
# for trimming cpus does not apply here, these 40 are genuinely used, and the
# sN=2500 benchmark OOMed inside umfpack_numeric! at solver_threads=4 with only
# 60G, so the memory headroom is the point.
#
# Unlike the earlier data_st4 pass this runs with BLAS pinned to one thread, so
# solver_threads is the only threading being tested rather than being read off
# against OpenBLAS contention.
#
# 60 rows 10 at a time is 6 waves, and maxtime is cut to 5h for this run, so 48h
# covers the worst case of 6 full waves with slack - the slowest row took a
# couple of hours even before the BLAS fix, so it should not come close. Finished
# rows are written individually and a resubmit skips them, so overrunning only
# loses what is in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main4_all_st4()'
