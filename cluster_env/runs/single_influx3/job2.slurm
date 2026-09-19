#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-9

# main2_B5: 20 Ks x 10 lis x 120 repeats = 24000 runs at B=5, an eighth of
# main1's 1500 cells. Sized for the 64039M nodes (compute-0-9/10/11) rather
# than the 96293M pair, so it can run alongside main1 on compute-0-7 without
# competing for the big nodes. 60G = 61440M leaves ~2.5G of the node's memory
# to the OS. main1 is at 8.4G peak for 7.5x the cells, so this has ample room.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2_B5()'
