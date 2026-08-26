#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 00:40:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-9
#SBATCH -o test2.out

# Same three configs on the same node class as job1, 3 minutes of solving each,
# into results_test - to prove the machinery works and measure the rate before
# committing to the 24h job.
#
# report_every=20 rather than the 300 job1 uses: at 300 a 180s run would print no
# progress line at all, which is the one thing this test is here to check.
time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1(; outdir="results_test", maxtime=180, report_every=20)'
