#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=34
#SBATCH --mem=44G

# Reduced-footprint twin of job1.slurm, sized to start immediately while
# xucapusa's re_fig5 array (53149) is still draining off the partition.
#
# Memory is the binding constraint, not cores - the array's tasks each reserve
# 8G. 44G fits on compute-0-7 today, 48G does not. gendata1 (B=3, 40 threads)
# peaked at 29.4G, so this leaves ~1.5x headroom for B=5.
#
# --time is generous: gendata1 took 2h35m on 40 threads.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'gendata2()'
