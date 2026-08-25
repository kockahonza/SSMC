#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 36:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=60G
#SBATCH -w compute-0-10

# main4_part2 runs a third of systems.jld2 - 10 of the 30 systems, ie 20 of the
# 60 rows - 10 at a time, one thread each. All three parts write into the same
# `data`: part_rows indexes into the full pde_df, so their row files cannot
# collide and nothing downstream needs to know about the split.
#
# Pinned to compute-0-10, one of the three free 64G nodes, so parts 1/2/3 land on
# 0-9/0-10/0-11 and the 96G compute-0-6 stays free. --mem is per node, not per
# cpu, and at 60G of 64G it still keeps anyone else off the node even though the
# cpu request no longer does.
#
# 12 cpus, not the 40 the main2_panel_e jobs took: BIOP sets no TRESBillingWeights,
# so fairshare is billed on *allocated* cpus whether or not they are used, and
# with run_threads=10 the other 28 were pure waste - job 53892 was charged 42 days
# of CPUTime for at most 10 days of real work. PriorityWeightFairShare is 10000,
# usage only half-lives (14d, no reset), and the account FairShare is already
# 0.005, so that overcharge comes back as queue position. 12 gives julia -t 12:
# the 10 row tasks, 2 spare for the main task and jldsave, and 6 GC threads
# (Julia 1.10 defaults gcthreads to half of -t), which these allocation-heavy
# solvers do use.
#
# 60G is deliberate over-provision: the equivalent main2_panel_e job (53892, also
# 10 rows at a time but at sN=2500) peaked at 24G RSS, and these run at sN=1000.
# The one thing that grew is save_step=10, 10x the saved states of that job - at
# 0.3M a state and its worst row's 39k steps that is ~1.2G for a bad row, so
# still far inside the budget.
#
# --time covers two 10h maxtime waves (20 rows / 10 at a time) plus slack. Rows
# that finish are written out individually and a resubmit skips them, so running
# out of time only loses the rows still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main4_part2()'
