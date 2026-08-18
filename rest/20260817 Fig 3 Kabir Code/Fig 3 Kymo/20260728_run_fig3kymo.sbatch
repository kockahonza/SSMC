#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 12:00:00
#SBATCH -N1
#SBATCH -n1
#SBATCH --cpus-per-task=3
#SBATCH -J fig3kymo

# Run the byproduct-diffusion kymographs (Consolidated/Fig 3 Kymo) on the cluster.
# One core per diffusion ratio, since the three runs go through a process pool.
# The defaults finish in minutes on a laptop; this is for reruns at a larger
# --ssize or --t-max, where the stiff solve grows quickly with both. The script
# writes 20260728_fig3kymo_data.npz and 20260728_fig3kymo_parameters.md next to
# itself, and the figures into Panels/.

# cd "/home/kabir/Work/03 UCL/Projects/2025 Spatial Ecology/20260609 Jans Code/SSMC-python/Consolidated/Fig 3 Kymo"

cd "/home/xucapusa/20260709 Spatial Ecology/Fig 3 Kymo"

eval "$(/home/xucapusa/miniconda3/bin/conda shell.bash hook)"
conda activate jupy

# Each run is a single stiff solve, which BLAS does not usefully parallelise;
# keep it from spawning threads that only contend between the workers.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun python 20260728_run_fig3kymo.py \
    --ps 0.01 0.1 1 \
    --t-max 1e6 \
    --ssize 1024 \
    --n-peaks 20
