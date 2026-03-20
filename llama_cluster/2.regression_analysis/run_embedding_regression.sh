#!/bin/bash
#SBATCH --job-name=text_regression
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=output.log

python textual_embedding_analysis.py
