#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 02:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=16G

# main6_on_Klp_point: single point (K=10, l=0.999, p=1), 3000 runs.
# ~0.033 s/run on 36 cores puts the solving at a couple of minutes; the output
# is ~220 MB of jld2 (73 kB/row), so 16G is ample.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main6_on_Klp_point()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/sid2_main6_on_Klp_point_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "si_disprel2 main6_on_Klp_point finished ok (job $SLURM_JOB_ID), main6_on_Klp_point/ is $(du -sh main6_on_Klp_point | cut -f1) over $(ls main6_on_Klp_point | wc -l) files" \
        --title "sid2 main6_on_Klp_point done" --attach "$tailfile"
else
    ~/notify.sh "si_disprel2 main6_on_Klp_point FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "sid2 main6_on_Klp_point failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
