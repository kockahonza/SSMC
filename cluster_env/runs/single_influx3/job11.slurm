#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-7

# main11_fs1_N10: second of the fixed-sparsity family (see job10.slurm), at
# N=M=10 - same size as main3_NM10, which this s=0.3 was chosen to reproduce
# the pe of. 60G on the 96G compute-0-7 is ample.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main11_fs1_N10()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/si3_main11_fs1_N10_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "single_influx3 main11_fs1_N10 finished ok (job $SLURM_JOB_ID), main11_fs1_N10.jld2 is $(du -h main11_fs1_N10.jld2 | cut -f1)" \
        --title "si3 main11_fs1_N10 done" --attach "$tailfile"
else
    ~/notify.sh "single_influx3 main11_fs1_N10 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "si3 main11_fs1_N10 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
