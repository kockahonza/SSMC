#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=90G

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main1()'
status=$?

# notify when done, with the tail of this job's log attached
log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/totbiom2_main1_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "si_pdes1 totbiom2 main1 finished ok (job $SLURM_JOB_ID), data1 is $(du -sh data1 | cut -f1)" \
        --title "totbiom2 main1 done" --attach "$tailfile"
else
    ~/notify.sh "si_pdes1 totbiom2 main1 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "totbiom2 main1 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
