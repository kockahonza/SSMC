#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G

# main2 is main1 with solver=TRBDF2 instead of QNDF - same 500 rows of
# systems.jld2, same tolerances, same 10h per row maxtime, 40 at a time with one
# thread each. It writes to data2, not data, so it cannot skip or overwrite the
# QNDF rows and the two solvers can be compared row by row.
#
# --mem=94G is not a memory requirement, it is the node filter: only
# compute-0-6/7 have 96G, so this lands on the same class of node as 53894, the
# QNDF run. That matters here because the point is a solver comparison and the
# two node classes are not the same machine - 0-6/7 are E5-2640 v4 at 2.4GHz,
# 0-9/10/11 are the older E5-2650 v3 at 2.3GHz. On actual usage 53894 peaked
# around 13.5G for all 40 rows, and TRBDF2 should sit at or below that: it is a
# one step method, so it carries none of QNDF's order-5 BDF history, while the
# sparse Jacobian and its UMFPACK factorization are the same size either way.
#
# Wall time is NOT comparable to 53894's 1d05:58 even on the same node: that run
# predates BLAS.set_num_threads(1) in ../base.jl, so it spent its time with 40
# OpenBLAS pools of 20 threads spinning against each other inside a 40 cpu
# cgroup. Any speedup seen here is that fix plus TRBDF2, not TRBDF2 alone. Step
# counts and final_T/retcode per row are the honest comparison; to get a clean
# timing number main1 would have to be rerun into a third directory with BLAS
# pinned.
#
# --time 72:00:00 matches job1. 53894 needed 1d06h with the BLAS contention and
# 12 of its 500 rows still ran out of iterations or time, so TRBDF2 could go
# either way - it takes cheaper steps but more of them, and the stiff rows are
# exactly where a 2nd order one step method struggles against BDF. Finished rows
# are written out individually and a resubmit skips them, so this is meant to be
# resubmitted until data2 is full.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2()'
