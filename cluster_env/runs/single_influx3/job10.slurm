#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-6

# main10_fs1_N5: first of the fixed-sparsity family (main10-13), same
# s=B/M=0.3 (matching main3_NM10's pe) at N=M=5,10,15,20 respectively. 30x10x120
# grid, 1.5x more K points than main1-9's 20x10x120, but N=5 gives the smallest
# state vectors of the four so this is the fastest. 60G on the 96G compute-0-6
# is ample given main2 (N=M=20) peaked at 7.3G.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main10_fs1_N5()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/si3_main10_fs1_N5_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "single_influx3 main10_fs1_N5 finished ok (job $SLURM_JOB_ID), main10_fs1_N5.jld2 is $(du -h main10_fs1_N5.jld2 | cut -f1)" \
        --title "si3 main10_fs1_N5 done" --attach "$tailfile"
else
    ~/notify.sh "single_influx3 main10_fs1_N5 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "si3 main10_fs1_N5 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
