#!/bin/bash
#
#SBATCH --job-name=user_stats
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/user_history_stats.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:0

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/
python ../user_history_stats.py \
--user_history_path "../data/spirs_history_timestamps/spirs-sarcastic-history.txt" \
--out_file "../data/spirs_history_timestamps/user_tweetscount.pkl" 
