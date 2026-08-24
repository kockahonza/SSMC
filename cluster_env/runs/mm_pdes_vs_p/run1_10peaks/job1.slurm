#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 02:00:00
#SBATCH --cpus-per-task=25
#SBATCH --mem=16G

# main1 runs all 50 rows two-per-core, one thread each, so 25 cores.
#
# Sized off a measured full-resolution row: ~10s of wall clock and ~8M of
# output. The MM is only 3 fields on 2500 gridpoints (7500 unknowns, 60K a
# state), so the whole 50 row sweep is minutes, not hours - unlike si_pdes1,
# whose 40-field SI systems needed 10h apiece. --time and --mem are therefore
# mostly slack: 2h against a measured ~1min, and 16G against a measured ~8M a
# row, the headroom being for the saved state series, which is unbounded by
# design (one state per 20 solver steps).
#
# main1 still caps each row at 1h of wall clock, so a row whose coarsening
# turns out to be far slower than the ones measured cannot eat the job.
# Rows that finish are written out individually and a resubmit skips them.
#
# setup1.jld2 must exist before this runs - make it once with
#   julia --project -L base.jl -e 'make_setup1()'
# and commit it, so the cluster uses the same initial conditions.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
