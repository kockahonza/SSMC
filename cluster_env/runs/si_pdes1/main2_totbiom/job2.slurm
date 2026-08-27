#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-9

# main2 runs the systems1.jld2 half (25 rows, 5 systems) 10 rows at a time,
# one thread each, writing to data1.
#
# Pinned to compute-0-9, one of the 64G nodes, to keep compute-0-6/0-7 (96G)
# free for other work. --mem is per node, ie for the whole job, not per cpu.
# ~30G of that is the 10 solvers (2.35G measured for one at sN=2000, scaled to
# the 2500 used here), the rest is headroom for the saved state series, which is
# unbounded by design - 781K a state, one per 100 steps, so a run taking 200k
# steps would hold 1.2G. The 18-at-a-time job1 gave each row 3.4G against the
# 5.3G main1_test_DN1e-6 actually used, and lost rows to OutOfMemoryError;
# 60G/10 is 6G a row.
#
# 40 cpus for 10 single-threaded rows looks wasteful but is not: --mem=60G on a
# 64G node has already taken everything anyone else could use the cores with,
# and the throughput this is sized against was measured at -t 40. solver_threads
# is off - see the note on main2/main3 in base.jl.
#
# --time covers three 10h maxtime waves (25 rows / 10 at a time) plus slack,
# rather than the one wave job1 budgeted for. Rows that finish are written out
# individually and a resubmit skips them, so running out of time only loses the
# rows still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2()'
