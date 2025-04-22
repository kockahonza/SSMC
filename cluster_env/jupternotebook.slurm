#!/bin/bash
#SBATCH --job-name=jupyter-notebook
#SBATCH -p BIOP
#SBATCH --time 36:00:00
#requesting 128 cpus
#SBATCH --cpus-per-task=128

module load Python

ssh -R 9999:localhost:9999 -Nf dias.hpc.phys.ucl.ac.uk

JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK jupyter notebook --no-browser --port 9999
