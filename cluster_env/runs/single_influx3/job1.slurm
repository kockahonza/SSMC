#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=92G
#SBATCH -w compute-0-6

# Node pin moved compute-0-9 -> compute-0-6. compute-0-9 has RealMemory=64039M,
# so --mem=92G (94208M) was unsatisfiable there and slurm rejected the job at
# submit with "Requested node configuration is not available". compute-0-6 and
# -0-7 are the 40-core/96293M pair, the same sizing single_influx2/job2.slurm
# settled on. compute-0-4/5 (515G) are still DOWN - not responding since
# 2026-06-25 - so 94G-ish remains the ceiling dias can offer.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
