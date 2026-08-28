#!/bin/bash
#SBATCH -J mm_run7
#SBATCH -p BIOP
#SBATCH -o run7_%j.out
#SBATCH --time 12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=90G

# One node, one julia process, 40 threads. make_setup1() writes setup1.jld2 and
# main1() then solves all 72 rows (6 ps x 12 reps), 40 at a time. The setup is
# regenerated here rather than shipped so the run maps to the committed base.jl;
# it is seeded, so it is the same file either way, and it throws immediately if
# any p turns out spatially stable - the cheapest way to fail this run early.
#
# One process rather than an array of one-row tasks: see run6's job_array.slurm
# for what happens when 50+ julia processes mmap the same precompile cache on
# the shared home filesystem at once.
#
# --exclusive plus --mem=90G takes a whole node and confines the job to
# compute-0-6/7, the two 96GB nodes. Memory is what is tight here, not cores:
# 40 rows are in flight at once, each holding its whole saved state series in
# memory until the row is written, and at sN=5000 with save_step=20 a long row
# can hold over a GB. The other 40 core nodes only have 64GB.
#
# --time is generous. 72 rows go in 2 waves of 40 and maxtime caps a row at 1h,
# so ~2h is expected. Rows are written one file each and a resubmit skips the
# ones already there, so an overrun only loses what is still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'make_setup1(); main1()'
