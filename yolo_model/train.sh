#!/bin/bash
#SBATCH --job-name=yolo_cubesat         # Name of your job
#SBATCH --output=training_output_%j.log # Standard output log
#SBATCH --error=training_error_%j.log   # Standard error log
#SBATCH --nodes=1                       # Run on a single node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=8               # Number of CPU cores for data loading
#SBATCH --mem=32G                       # Amount of RAM requested
#SBATCH --gres=gpu:mi210:1              # Request exactly 1 AMD MI210 GPU
#SBATCH --time=24:00:00                 # Maximum time limit (HH:MM:SS)
#SBATCH --partition=muscadine           # The specific cluster partition

echo "Job started at: $(date)"
echo "Running on node: $SLURM_NODELIST"

# 1. Load the required ROCm library
module load rocm-smi-lib

# 2. Navigate to your working directory
cd /work/users/ng739/cubesat/model/cubesat/yolo_model

# 3. Verify the GPU is visible to the compute node
echo "--- GPU Status ---"
rocm-smi --showproductname

# 4. Run the training script
echo "--- Starting YOLO Training ---"
python3 -u main.py train

echo "Job finished at: $(date)"
