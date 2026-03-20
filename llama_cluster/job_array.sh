#!/bin/bash
#PBS -N llama_array
#PBS -q normal
#PBS -l nodes=1:ppn=128
#PBS -l mem=180gb
#PBS -l walltime=06:00:00
#PBS -J 1-4

cd /home/yl9210a-hpc/chem_project/llama
source /home/yl9210a-hpc/chem_project/.venv/bin/activate

python run_llama_parallel.py $PBS_ARRAY_INDEX 4
