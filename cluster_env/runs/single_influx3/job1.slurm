#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=92G

# No -w: the old pin to compute-0-9 was unsatisfiable, since that node has
# RealMemory=64039M and --mem=92G is 94208M, so slurm bounced the job at submit
# with "Requested node configuration is not available". Leaving the node to the
# scheduler, 40 cores + 92G already narrows BIOP to compute-0-6/-0-7 (the
# 40-core/96293M pair, the sizing single_influx2/job2.slurm settled on); the
# other idle BIOP nodes are 40-core/64039M or 8-core. compute-0-4/5 (515G) are
# still DOWN - not responding since 2026-06-25 - so 94G-ish is the dias ceiling.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
