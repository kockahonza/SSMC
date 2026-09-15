#!/bin/bash
#SBATCH -p BIOP
#SBATCH --time 01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH -o reduce_stats.out
julia -t 1 --project -L base.jl reduce_stats.jl
