#!/bin/bash
#SBATCH --job-name=sturctual_regression
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=output.log

python structual_embedding_analysis.py
