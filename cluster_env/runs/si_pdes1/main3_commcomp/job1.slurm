#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G

# main1 runs all 500 rows of systems.jld2 (100 systems x 5 ps) 40 at a time,
# one thread each, writing to data/.
#
# --mem=94G is all of a 96G node (RealMemory is 96293M) and doubles as the node
# filter: only compute-0-6/7 have that much, so Slurm cannot drop this on one of
# the 64G nodes the way a bare -c 40 would. No -w, so it takes whichever of the
# two is free; add one to pin it.
#
# 40 rows against 94G is a wide margin. jobbench (53889) held 23.5G for 10
# concurrent rows at sN=2500 with the state series on, ie ~2.2G a row over a
# ~1.5G Julia baseline. Both of those costs are much smaller here: sN is 250, a
# tenth, and the solver's memory (sparse Jacobian plus its factorisation) is
# linear in sN, so ~0.25G a row; and save_step=nothing drops the saved state
# series entirely, which was the unbounded part and what took job1 of
# main2_totbiom to 55.8G. That puts the real figure near 12G. Even pricing every
# row at the full sN=2500 cost and ignoring both savings gives ~90G, which still
# fits.
#
# 500 rows 40 at a time is 13 waves, so --time is well short of the 130h worst
# case where every row runs to its 10h maxtime. That is deliberate - finished
# rows are written out individually and a resubmit skips them, so this is meant
# to be resubmitted until data/ is full. At sN=250 rows should be far quicker
# than the cap and it may well not come to that.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
