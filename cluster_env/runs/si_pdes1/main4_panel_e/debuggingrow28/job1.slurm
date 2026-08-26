#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-9

# main1 runs the three configs from one doubled-grid state, concurrently, 13
# solver threads each - 39 of the node's 40 cores genuinely used, so unlike the
# production jobs the full cpu request is not fairshare waste.
#
# On a 64G node (compute-0-9/10/11, 64039M) rather than a 96G one, to leave those
# free for production runs. save_step=50 is ~4.6G of saved series per config over
# a 20h run, so three of those plus three factorisations sits well inside 60G.
#
# Each config writes its own file the moment it finishes and a failure in one is
# caught, so a config that blows up costs only itself.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
