#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G

# --mem is per node, ie for the whole job, not per cpu. main1 runs its 9 rows
# all at once, so they share it: ~14G each, against ~3G measured for one run at
# sN=2500 and the ~3.9G per run pdes6 got by on. Deliberately over.
#
# 9 rows at 4 solver threads each uses 36 of the 40 cores.
#
# --time is one 10h maxtime wave plus slack. Rows that finish are written out
# individually and a resubmit skips them, so running out of time only loses the
# rows still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
