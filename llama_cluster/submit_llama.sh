#!/bin/bash
#PBS -N llama_parallel_4
#PBS -q normal
#PBS -l nodes=1:ppn=128
#PBS -l mem=180gb
#PBS -l walltime=06:00:00

# ⭐ Key：Force into directory
BASE_DIR="/home/yl9210a-hpc/chem_project/llama"
cd $BASE_DIR

# create folders
mkdir -p $BASE_DIR/logs
mkdir -p $BASE_DIR/output

# Set Log path (PBS command)
#PBS -o /home/yl9210a-hpc/chem_project/llama/logs/llama_all.out
#PBS -e /home/yl9210a-hpc/chem_project/llama/logs/llama_all.err

source /home/yl9210a-hpc/chem_project/.venv/bin/activate

echo "[*] Starting 4-way parallel inference..."

# Use absolute path to execute Python Script
python $BASE_DIR/0.run_llama_parallel.py 1 4 > $BASE_DIR/logs/part1.log 2>&1 &
python $BASE_DIR/0.run_llama_parallel.py 2 4 > $BASE_DIR/logs/part2.log 2>&1 &
python $BASE_DIR/0.run_llama_parallel.py 3 4 > $BASE_DIR/logs/part3.log 2>&1 &
python $BASE_DIR/0.run_llama_parallel.py 4 4 > $BASE_DIR/logs/part4.log 2>&1 &

# Wait till all the files finish running
echo "[*] Waiting for all parallel tasks to finish..."
wait

# 3. check the output and run the combine script
echo "[*] Parallel tasks finished. Starting combination script..."
python $BASE_DIR/1.combine.py > $BASE_DIR/logs/combine.log 2>&1

echo "[*] All steps finished! Final output is ready."

# BGE-M3 在 128 核下跑 1500 筆大約只需要 5-10 分鐘
# python $BASE_DIR/2.run_embedding.py

# echo "[*] MISSION COMPLETE! All molecules processed and embedded."