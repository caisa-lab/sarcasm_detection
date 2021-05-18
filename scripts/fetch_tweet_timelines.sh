#!/bin/bash
#
#SBATCH --job-name=fetching
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/fetch.txt
#SBATCH --mail-user=joanplepi@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --partition=cai
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:0

source /ukp-storage-1/plepi/anaconda3/etc/profile.d/conda.sh
conda activate spirs
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/ukp-storage-1/plepi/anaconda3/lib/
python ../fetch_tweet_timelines.py \
--in_file "data/spirs_history/SPIRS-sarcastic-history.txt" \
--out_file "data/spirs_history/SPIRS-sarcastic-history-extended.txt" \
--start_line 4241594


#1243399
