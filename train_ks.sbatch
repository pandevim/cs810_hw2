#!/bin/bash
#SBATCH --job-name=train_ks
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_ks_%j.out
#SBATCH --error=logs/train_ks_%j.err

mkdir -p logs

# Load the python module
module purge
module load python39

# Create and activate a virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install h5py numpy scipy huggingface_hub wandb matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cu121

export HF_TOKEN="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"
export WANDB_API_KEY="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"

python -u train_ks.py
