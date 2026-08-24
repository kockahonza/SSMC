#!/bin/bash
#SBATCH -J mm_run6
#SBATCH -p BIOP
#SBATCH -o run6_%j.out
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=48G

# One node, one julia process, 40 threads - main1() runs 25 rows at a time
# inside it.
#
# NOT an array of one-row tasks, which is what was tried first (job_array.*)
# and does not work here: 50+ julia processes starting at once all mmap the
# same precompile cache on the shared home filesystem, and they die with
# "protect_page: Invalid argument" during JIT setup or a plain Bus error a
# minute in. 82 of 112 tasks failed that way and not one row was written. A
# single process has none of that contention.
#
# --mem is for the whole job: 25 concurrent rows, each holding a saved state
# series that is ~180MB at worst (sN=5000, save_step=200), plus julia itself.
#
# --time is generous because the p=2 rows are the slow ones and maxtime in
# main1 caps a single row at 6h. Rows that finish are written individually and
# a resubmit skips them, so overrunning only loses what is still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
