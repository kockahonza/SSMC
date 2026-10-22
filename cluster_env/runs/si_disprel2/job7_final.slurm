#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 04:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G

# main7_final: K=10, l=0.999, p in (0.01, 0.1, 1), 1000 runs each.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main7_final()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/sid2_main7_final_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "si_disprel2 main7_final finished ok (job $SLURM_JOB_ID), main7_final/ is $(du -sh main7_final | cut -f1) over $(ls main7_final | wc -l) files" \
        --title "sid2 main7_final done" --attach "$tailfile"
else
    ~/notify.sh "si_disprel2 main7_final FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "sid2 main7_final failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
