#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=94G

# Whole-node twin of job1.slurm, sized to fill compute-0-6 (40 cores,
# RealMemory=96293M). 94G = 96256M leaves the node's 37M slack, so this takes
# all the cores and effectively all the memory.
#
# Supersedes the 34-core/44G sizing, which was a deliberate squeeze to start
# alongside xucapusa's re_fig5 array while it drained. That cap is what killed
# job 53887: it finished the compute and then died in jldsave with
# "SystemError: mmap: Cannot allocate memory" - JLD2's MmapIO has to map the
# growing output file on top of the ~28G of raw_dfs still resident.
#
# Sizing is uncertain, not comfortable. base.jl now runs the full 30 leakage
# values (the lis = [1., 0.999, ...] override is gone), so this is ~6x the work
# and ~6x the output of 53887. The closest completed reference is single_influx
# gd5: same 30 x 50 x 100 shape at B=3, which peaked at 68.7G and wrote 11G.
# This is B=5, so 94G may still not be enough. compute-0-4/5 (512G) are DOWN,
# so 94G is the most dias can offer today.
#
# --time raised 24h -> 72h for the same reason: gendata1's 2h35m is no longer a
# useful reference. gd5 took 7h45m on 102 cores; this has 40.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'gendata2()'
