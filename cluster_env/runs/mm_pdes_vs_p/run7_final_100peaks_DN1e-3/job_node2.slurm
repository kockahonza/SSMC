#!/bin/bash
#SBATCH -J mm_run7_T1e7
#SBATCH -p BIOP
#SBATCH -o run7_main2_%j.out
#SBATCH --time 6:00:00
#SBATCH --cpus-per-task=40
#SBATCH --exclusive
#SBATCH --mem=60G

# job_node.slurm with T cut from 1e12 to 1e7: make_setup2() writes setup2.jld2
# and main2() solves the same 72 rows into main2/. Different setup file and
# different output directory, so this can run at the same time as the main1 job
# on another node without the two colliding.
#
# --mem=60G rather than job_node.slurm's 90G so this fits the 64GB nodes
# (compute-0-9/10/11) as well as the 96GB pair, since main1 is likely occupying
# one of those. It is not a tight budget: the main1 job peaked at 15.7GB with
# all 40 rows in flight, and these rows stop 5 decades earlier, so they hold
# far shorter state series.
#
# --time 6h is loose. The T=1e12 rows that reached p=0.562 took ~20 min each and
# main1 was ~40 rows an hour; stopping at 1e7 should be well inside that.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'make_setup2(); main2()'
