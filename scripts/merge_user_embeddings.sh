#!/bin/bash
#
#SBATCH --job-name=nonsar_user_embeddings
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/merge_emb.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:0

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/

python ../merge_embeddings.py \
--data_dir "../data/spirs_history_timestamps" \
--out_path "../data" 