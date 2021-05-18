#!/bin/bash
#
#SBATCH --job-name=link
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/setup.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=1GB
#SBATCH --gres=gpu:0

sma_toolkit="/ukp-storage-1/plepi/sma_toolkit/"

#link toolkit and embeddings
mkdir sma_toolkit
ln -s ${sma_toolkit} ../sma_toolkit

