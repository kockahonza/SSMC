#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G

# main13_fs1_N20: last of the fixed-sparsity family (see job10.slurm), at
# N=M=20 - the same size as main1/main2's default, so the largest matrices of
# the four and the most comparable in per-run cost to main2's 3h14m for a
# 20x10x120 grid; at 30x10x120 (1.5x the K points) this could plausibly want
# ~5h, well inside the 72h limit. No -w pin: compute-0-6/7/9 are claimed by
# job10-12 and compute-0-10/11 are still running job8/9, so this is left to
# schedule on whichever 40-core BIOP node frees first.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main13_fs1_N20()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/si3_main13_fs1_N20_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "single_influx3 main13_fs1_N20 finished ok (job $SLURM_JOB_ID), main13_fs1_N20.jld2 is $(du -h main13_fs1_N20.jld2 | cut -f1)" \
        --title "si3 main13_fs1_N20 done" --attach "$tailfile"
else
    ~/notify.sh "single_influx3 main13_fs1_N20 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "si3 main13_fs1_N20 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
