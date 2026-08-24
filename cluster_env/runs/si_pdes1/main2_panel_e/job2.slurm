#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G

# main2 runs the systems1.jld2 half (25 rows, 5 systems) 10 rows at a time,
# each solver given the leftover cores via solver_threads=4: 10 x 4 = 40.
#
# Sized to take a whole 64G node. --mem is per node, ie for the whole job, not
# per cpu. ~30G of that is the 10 solvers (2.35G measured for one at sN=2000,
# scaled to the 2500 used here), the rest is headroom for the saved state
# series, which is unbounded by design - 781K a state, one per 100 steps, so a
# run taking 200k steps would hold 1.2G. The 18-at-a-time job1 gave each row
# 3.4G against the 5.3G main1_test_DN1e-6 actually used, and lost rows to
# OutOfMemoryError; 60G/10 is 6G a row.
#
# --time covers three 10h maxtime waves (25 rows / 10 at a time) plus slack,
# rather than the one wave job1 budgeted for. Rows that finish are written out
# individually and a resubmit skips them, so running out of time only loses the
# rows still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2()'
