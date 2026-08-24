#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 12:00:00
#SBATCH --cpus-per-task=9
#SBATCH --mem=48G

# main1 runs its 9 rows at once, one thread each, so 9 cores.
#
# --mem is per node, ie for the whole job, not per cpu. ~27G of that is the 9
# solvers (2.35G measured for one at sN=2000, scaled to 2500), the rest is
# headroom for the saved state series, which is unbounded by design - 781K a
# state, one per 100 steps, so a run taking 200k steps would hold 1.2G.
#
# --time is one 10h maxtime wave plus slack. Rows that finish are written out
# individually and a resubmit skips them, so running out of time only loses the
# rows still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
