#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 24:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=24G

# main5_pd_cov1: 20 K x 10 li x 3 p = 600 points at 100 runs each, so 60k
# systems and ~4.4 GB of uncompressed jld2 (73 kB/row, nearly all of it the
# 20x20x20 D array in params).
#
# Timing: main1 did 1500 runs in 49 s on 36 Myriad cores with this same
# solve_si_odes (T=1e8, N=M=20), ~0.033 s/run, with no K trend over K=3..100 -
# that puts the solving here at ~30 min. main4's per-file gaps suggest the JLD2
# writes cost more than the ODEs do, so a few hours is the realistic figure.
# K now goes to 1e4, two decades past anything main1 sampled, so the ceiling is
# the maxtime=30 cap on every solve: 60000*30/40 = 12.5 h. 24 h is slack on top.
#
# Memory is small - one 100-row df (~7.3 MB) is live at a time - so 24G is
# plenty and keeps this off the 64 GB nodes' limit. BIOP is fully idle, so no
# -w pin; any of compute-0-6/7/9/10/11 will do.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main5_pd_cov1()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/sid2_main5_pd_cov1_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "si_disprel2 main5_pd_cov1 finished ok (job $SLURM_JOB_ID), main5_pd_cov1/ is $(du -sh main5_pd_cov1 | cut -f1) over $(ls main5_pd_cov1 | wc -l) files" \
        --title "sid2 main5_pd_cov1 done" --attach "$tailfile"
else
    ~/notify.sh "si_disprel2 main5_pd_cov1 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "sid2 main5_pd_cov1 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
