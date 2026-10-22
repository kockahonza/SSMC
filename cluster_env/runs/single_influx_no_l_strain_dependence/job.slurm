#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem=60G

# main2: N=M=20, B=3, byproducts set by resource not strain
# (JansSampler3_resdep_byproducts), 50 Ks x 4 lis x 120 repeats.

time julia -t $SLURM_CPUS_PER_TASK --project -L base.jl -e 'main2()'
status=$?

log="slurm-${SLURM_JOB_ID}.out"
tailfile="/tmp/sinolsd_main2_${SLURM_JOB_ID}_tail.txt"
tr '\r' '\n' < "$log" | tail -40 > "$tailfile"
if [ $status -eq 0 ]; then
    ~/notify.sh "single_influx_no_l_strain_dependence main2 finished ok (job $SLURM_JOB_ID), main2.jld2 is $(du -h main2.jld2 | cut -f1)" \
        --title "si_no_l_strain_dep main2 done" --attach "$tailfile"
else
    ~/notify.sh "single_influx_no_l_strain_dependence main2 FAILED with exit code $status (job $SLURM_JOB_ID)" \
        --title "si_no_l_strain_dep main2 failed" --attach "$tailfile"
fi
rm -f "$tailfile"

exit $status
