#!/bin/bash
#SBATCH -J main8_mms
#SBATCH -p BIOP
#SBATCH --time 12:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=40G

# Fits (seconds) then all 80 MM PDE rows into main1/, 40 at a time. Submit from
# this directory. make_fits_and_setup refuses to overwrite fits.jld2/setup.jld2,
# so for a resubmit drop it from the -e and run only main1(), which skips
# finished rows.

time julia -t $SLURM_CPUS_PER_TASK --project -L run_corresponding_mms.jl -e 'make_fits_and_setup(); main1()'
status=$?

# notify when done, with the tail of this job's log attached
log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/main8_mms_main1_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "si_pdes1 main8 corresponding_mms main1 finished ok (job $SLURM_JOB_ID), $(ls main1 | grep -c '^row') row files, $(du -sh main1 | cut -f1)" \
        --title "main8 MMs done" --attach "$tailfile"
else
    ~/notify.sh "si_pdes1 main8 corresponding_mms main1 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "main8 MMs failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
