#!/bin/bash
#
#SBATCH --job-name=temp_user_embeddings
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/temp_user_embeddings.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/
python ../user_embeddings.py \
--user_history_path "../data/spirs_history_timestamps/temp.txt" \
--out_file "../data/spirs_history_timestamps/temp_user_embeddings.txt" 
