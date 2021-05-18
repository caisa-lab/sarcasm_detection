#!/bin/bash
#
#SBATCH --job-name=gat_classification
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/gat_classification.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/

python ../gat_train.py \
--model_dir='/ukp-storage-1/plepi/users_sarcasm/data/gatClassify/' \
--data_dir='../data/' \
--config_dir='../configs' \
--user_mentions_file='user_mentions2.json' \
--user_embeddings_path='../data/SPIRS_500_neg15.txt'
