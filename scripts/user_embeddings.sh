#!/bin/bash
#
#SBATCH --job-name=user_embeddings
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/user_embeddings.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/

python ../user_embeddings.py \
--user_history_path "/ukp-storage-1/plepi/usr2vec/raw_data/pruned_filtered_spirs_history.txt" \
--out_file "../data/spirs_history_timestamps/avg768Paraphrase_user_embeddings.txt" \
--bert_model "paraphrase-distilroberta-base-v1"

#--user_history_path "../data/spirs_history_timestamps/spirs-non-sarcastic-filtered-history.txt" \
#--out_file "../data/spirs_history_timestamps/filtered_user_embeddings.txt" 
