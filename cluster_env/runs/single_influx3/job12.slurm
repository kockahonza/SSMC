#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-9

# main12_fs1_N15: third of the fixed-sparsity family (see job10.slurm), at
# N=M=15. 60G on the 64G compute-0-9 matches the sizing used for main5-9,
# which ran N=M=20 (larger state vectors) fine at the same mem.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main12_fs1_N15()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/si3_main12_fs1_N15_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "single_influx3 main12_fs1_N15 finished ok (job $SLURM_JOB_ID), main12_fs1_N15.jld2 is $(du -h main12_fs1_N15.jld2 | cut -f1)" \
        --title "si3 main12_fs1_N15 done" --attach "$tailfile"
else
    ~/notify.sh "single_influx3 main12_fs1_N15 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "si3 main12_fs1_N15 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
