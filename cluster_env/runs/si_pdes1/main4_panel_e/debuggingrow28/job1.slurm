#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G
#SBATCH -w compute-0-6

# main1 runs the three configs from one doubled-grid state, concurrently, 13
# solver threads each - 39 of the node's 40 cores genuinely used, so unlike the
# production jobs the full cpu request is not fairshare waste.
#
# All of a 96G node. sN=2000 doubles the unknowns against the production runs, so
# each solver's factorisation is well over the 2.35G measured at sN=2500, and the
# saved series is 640K a state: at save_step=20 a run taking 200k steps holds 6G
# on its own. Three of those plus three factorisations is the reason for 94G
# rather than 60G - memory is the thing that would kill this, not cores.
#
# Each config writes its own file the moment it finishes and a failure in one is
# caught, so a config that blows up costs only itself.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
