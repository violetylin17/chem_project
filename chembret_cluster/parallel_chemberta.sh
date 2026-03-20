#!/bin/bash
#PBS -N chembreta_parallel_4
#PBS -q normal
#PBS -l nodes=1:ppn=128
#PBS -l mem=180gb
#PBS -l walltime=06:00:00
#PBS -j oe
#PBS -o /home/yl9210a-hpc/chem_project/chembret_cluster/chembreta_all.out

BASE_DIR="/home/yl9210a-hpc/chem_project/chembret_cluster"
cd $BASE_DIR

mkdir -p logs
mkdir -p output_chembret

source /home/yl9210a-hpc/chem_project/.venv/bin/activate

export PYTHONUNBUFFERED=1

echo "[*] Starting 4-way ChemBERTa parallel embedding (BACE mol -> embedding)..."

TOTAL_PARTS=4

> logs/part1.log
> logs/part2.log
> logs/part3.log
> logs/part4.log
> logs/combine.log

python bace_smiles_to_chembreta_embeddings.py 1 $TOTAL_PARTS > logs/part1.log 2>&1 &
python bace_smiles_to_chembreta_embeddings.py 2 $TOTAL_PARTS > logs/part2.log 2>&1 &
python bace_smiles_to_chembreta_embeddings.py 3 $TOTAL_PARTS > logs/part3.log 2>&1 &
python bace_smiles_to_chembreta_embeddings.py 4 $TOTAL_PARTS > logs/part4.log 2>&1 &
echo "[*] Waiting for all tasks..."
wait

echo "[*] Combining results..."
python bace_smiles_to_chembreta_embeddings.py --combine $TOTAL_PARTS > logs/combine.log 2>&1

echo "[*] ALL DONE!"
