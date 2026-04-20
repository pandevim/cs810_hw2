#!/bin/bash
#SBATCH --job-name=generate_ks
#SBATCH --partition=compute-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/generate_ks_%j.out
#SBATCH --error=logs/generate_ks_%j.err

mkdir -p logs

# Load the python module
module load python/3.11.10

# Create and activate a virtual environment
if [ ! -d "venv" ]; then
    python -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip
pip install h5py numpy huggingface_hub wandb

export HF_TOKEN="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"
export WANDB_API_KEY="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"

srun python -u generate_ks.py
