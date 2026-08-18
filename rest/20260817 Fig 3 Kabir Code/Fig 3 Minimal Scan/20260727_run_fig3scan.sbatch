#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 72:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=36
#SBATCH -J fig3scan

# Scan the byproduct mobility p = D_x/D_ext with replicates (Consolidated/Fig. 3
# Minimal Scan). The scan shape is passed explicitly on the srun line below, so this
# file is the single visible statement of what the job runs.
#
# Every run is one independent single-threaded stiff solve, so the scan scales with
# cores; the script sizes its worker pool to the cores the allocation actually grants,
# making --cpus-per-task above the only place the core count needs setting. With fewer
# cores than runs the remainder goes in a second wave, whose length is set by the slow
# runs near p = 1 where the pattern coarsens the most.
#
# The npz is rewritten after every completed run, so a job that hits the wall limit
# still leaves every finished replicate on disk. Re-running with --plot-only redraws
# both figures from that file without solving.
#
# Writes 20260727_fig3scan_peaks.npz and the scan figures next to itself.

# cd "/home/kabir/Work/03 UCL/Projects/2025 Spatial Ecology/20260609 Jans Code/SSMC-python/Consolidated/Fig. 3 Minimal Scan"

cd "/home/xucapusa/20260709 Spatial Ecology/Fig 3 Minimal Scan"

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

# Each worker is single-threaded by design; keep BLAS from spawning threads that only
# contend with the other workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python 20260727_run_fig3scan.py \
    --ps 0.05 0.1 0.25 0.5 1 2 \
    --replicates 10 \
    --t-max 1e8 \
    --ssize 2048 \
    --n-peaks 100
