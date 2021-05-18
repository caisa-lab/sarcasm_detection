#!/bin/bash
#
#SBATCH --job-name=bert_classification
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/train_bert.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/
python ../bert_users_training.py \
--model_dir='/ukp-storage-1/plepi/users_sarcasm/data/bert/' \
--data_dir='../data/' \
--config_dir='../configs' \
--task='perspective'
