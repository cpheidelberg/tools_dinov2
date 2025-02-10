#!/bin/bash

#SBATCH --job-name=trainViT       # Name des Jobs
#SBATCH --output=train-%j.out      # Standardausgabe und -fehler (j steht für Job-ID)
#SBATCH --error=train-%j.err       # Standardfehler (optional, wenn getrennt von stdout gewünscht)
#SBATCH --time=72:00:00            # Maximale Laufzeit (hier: 1 Stunde)
#SBATCH --partition=gpu-single            # Job-Partition
#SBATCH --ntasks=1                 # Anzahl der Tasks (hier: 1)
#SBATCH --cpus-per-task=12          # CPUs pro Task
#SBATCH --mem=64GB                 # Speicher pro Node
#SBATCH --gres=gpu:A100:1		   # Anzahl der GPUs

# Module laden
module load devel/miniconda/3
module load devel/cuda/12.2
module load lib/hdf5

# Variables
REPO_URL="https://github.com/cpheidelberg/tools_dinov2"  # Repository URL
REPO_DIR="tools_dinov2"  # Repository directory name
ENV_NAME="dinov2"  # Name of the conda environment
REQUIREMENTS_FILE="conda.yaml"  # Path to the conda requirements file
CONFIG_FILE="ssl_default_config.yaml"  # Path to the configuration file
#INPUT_DIR=/home/hd/hd_hd/hd_na236/all_png    /mnt/sds-hd/sd19g003/Glomeruli_dataset/WSI_patches/WSI_patches_test_5.h5
INPUT_DIR="/mnt/sds-hd/sd19g003/Glomeruli_dataset/WSI_patches_curatedpng/glommeruli_0.999" 
OUTPUT_DIR="/home/hd/hd_hd/hd_na236/checkpoints_folder/checkpoints_run_33"
PRE_WEIGHTS="/home/na236/pytorch_model_uni.bin"

# Clone the repository if it doesn't exist
if [ -d "$REPO_DIR" ]; then
    echo "Repository '$REPO_DIR' already exists. Skipping cloning."
else
    echo "Cloning repository from '$REPO_URL'..."
    git clone "$REPO_URL"
fi

# Navigate to the repository directory
cd "$REPO_DIR" || { echo "Directory '$REPO_DIR' not found."; exit 1; }

# Initialize conda
conda init

source activate dinov2

export WANDB_API_KEY = 0bd97b3595d946e185eef019c62eb95b4a3f4916

# Check if the conda environment exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists. Skipping environment creation."
else
     echo "Creating conda environment '$ENV_NAME' from '$REQUIREMENTS_FILE'..."
    conda env create -f "$REQUIREMENTS_FILE"
fi

cp "$CONFIG_FILE" "$OUTPUT_DIR"
cp "/home/hd/hd_hd/hd_na236/run.sh" "$OUTPUT_DIR"

# Execute the Python training script
echo "Running the training script with configuration file '$CONFIG_FILE'..."
python3 dinov2/train/train.py --config-file "$CONFIG_FILE" --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR" --pretrained-weights "$PRE_WEIGHTS"

echo "Script completed."

