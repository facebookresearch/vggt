#!/usr/bin/env bash
#SBATCH -A NAISS2024-5-609
#SBATCH -o /mimer/NOBACKUP/groups/snic2022-6-266/davnords/vggt/evaluation/slurm_outs/%x_%j.out
#SBATCH -t 0-02:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes 1

python evaluation/dtu.py