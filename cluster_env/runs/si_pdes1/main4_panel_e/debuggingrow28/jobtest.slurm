#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 00:40:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G
#SBATCH -w compute-0-6
#SBATCH -o test.out

# Same three configs, 3 minutes of solving each, into results_test - just to prove
# the machinery works and to measure the rate before committing to the 24h job.
time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1(; outdir="results_test", maxtime=180)'
