#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH -p BIOP
#SBATCH --time 36:00:00
# jaja
#SBATCH --cpus-per-task=32

module load Python

ssh -R 10000:localhost:10000 -Nf dias.hpc.phys.ucl.ac.uk

JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK jupyter notebook --no-browser --port 10000
