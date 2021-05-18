#!/bin/bash
#
#SBATCH --job-name=gat_classification
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/debug_gat_classification_1.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=testing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/
python ../gat_train.py \
--model_dir='/ukp-storage-1/plepi/users_sarcasm/data/gatClassify/' \
--data_dir='../data/' \
--config_dir='../configs'
