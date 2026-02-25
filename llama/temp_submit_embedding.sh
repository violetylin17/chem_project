#!/bin/bash
#PBS -N bge_embedding
#PBS -q normal
#PBS -l nodes=1:ppn=128
#PBS -l mem=100gb
#PBS -l walltime=01:00:00

BASE_DIR="/home/yl9210a-hpc/chem_project/llama"
cd $BASE_DIR

source /home/yl9210a-hpc/chem_project/.venv/bin/activate

echo "[*] Starting BGE-M3 Embedding..."
# 執行你寫好的 Embedding 腳本
python $BASE_DIR/2.embedding.py > $BASE_DIR/logs/embedding.log 2>&1

echo "[*] Embedding process finished."