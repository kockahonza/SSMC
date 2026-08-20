#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 12:00:00
#SBATCH --cpus-per-task=27
#SBATCH --mem=48G

# Sized to start now rather than to fill a node. BIOP's two 128 core/515G nodes
# are down, leaving 40 core nodes with 64G (compute-0-9/10/11) or 96G
# (compute-0-6/7), all partly occupied - 29-33 cores free at the time of
# writing. So 9 rows x 3 solver threads = 27 cores fits any of them.
#
# --mem is per node, ie for the whole job, not per cpu, so the 9 concurrent
# runs share it: ~5G each against ~3G measured for one run at sN=2500 and the
# ~3.9G per run pdes6 got by on. Still over, but within a 64G node.
#
# --time is one 10h maxtime wave plus slack. Rows that finish are written out
# individually and a resubmit skips them, so running out of time only loses the
# rows still in flight.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
