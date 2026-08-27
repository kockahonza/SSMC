#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 11:00:00
#SBATCH --cpus-per-task=11
#SBATCH --mem=16G

# Downsized greedy twin of job1.slurm, run alongside 53894 as a safeguard.
#
# 53894 has 489/500 rows and its last 11 (68-78) all sit in one chunk of the
# default :dynamic scheduler, so it is walking them one at a time. run_pde_setup
# now uses scheduler=:greedy, where run_threads is a worker pool off a shared
# queue, so this picks up all 11 remaining rows at once.
#
# run_pde_setup skips rows whose row%04d.jld2 already exists, so this only runs
# what 53894 has not finished. Both jobs write into "data"; whichever produces a
# row first wins and the duplicated work is deliberate.
#
# 11 cores / 16G rather than a whole node: 53894 peaked at 13.5G with up to 40
# rows in flight and save_step=nothing keeps each row small, so 16G for 11 rows
# is generous. --time covers one full maxtime=10h row plus startup.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'run_pde_setup("systems.jld2", "data"; save_step=nothing, abstol=1e-7, reltol=1e-9, maxtime=10 * 60 * 60, run_threads=Threads.nthreads(), solver_threads=nothing)'
