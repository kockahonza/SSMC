#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G
#SBATCH -w compute-0-11

# main9_B4: same 20 x 10 x 120 grid as job2/job3/job4 but at B=4 instead of
# main2's B=5. Same 60G sizing on the 64039M nodes for the same reason.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main9_B4()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/si3_main9_B4_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "single_influx3 main9_B4 finished ok (job $SLURM_JOB_ID), main9_B4.jld2 is $(du -h main9_B4.jld2 | cut -f1)" \
        --title "si3 main9_B4 done" --attach "$tailfile"
else
    ~/notify.sh "single_influx3 main9_B4 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "si3 main9_B4 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
