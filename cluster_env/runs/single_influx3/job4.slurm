#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 7-00:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-9

# main4_NM50: same 20 x 10 x 120 grid as job2/job3 but at N=M=50, so the state
# vectors are 100 long against the default 40. Memory is not the constraint -
# stored columns scale with N+M and main3 (N=M=10) peaked at 5.9G, main2 at
# 7.3G - so 60G on a 64039M node is still ample.
#
# TIME is the constraint. The stability scan does 2001 eigvals per surviving
# run, and eigvals is O(n^3): 100x100 is ~15x the cost of the 40x40 matrices
# main1/main2 worked with. main2 took 3h14m for this grid shape, so main4 could
# plausibly want 30-60h. BIOP has MaxTime=UNLIMITED, hence 7 days rather than
# the 72h used elsewhere - a limit that truncates a multi-day run is worse than
# one that is never reached.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main4_NM50()'
