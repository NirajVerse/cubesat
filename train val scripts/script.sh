#!/bin/bash
#SBATCH --job-name=cubesat_model        # Name of your job
#SBATCH --partition=muscadine           # The partition you have access to
##SBATCH --nodelist=muscadine-node-1     # Force it to the only node with a GPU!
##SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH --time=12:00:00                 # Maximum time to run (HH:MM:SS)
#SBATCH --output=train_log_%j.txt       # Saves standard output (the %j adds the job ID)
#SBATCH --error=train_error_%j.txt      # Saves crash/error messages

# 1. Load the GPU drivers
module load rocm/6.2.4


# 3. Activate your specific Python environment
source /work/users/ng739/cubesat/cubesat/bin/activate

# 4. Navigate to your code folder
cd "/work/users/ng739/cubesat/model/cubesat/train val scripts"

# 5. Run the training script! (Change 'two_class.py' if your main file is named differently)
python3 main.py
