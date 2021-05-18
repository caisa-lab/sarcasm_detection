#!/bin/bash
#
#SBATCH --job-name=filter_duplicates
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/filter_duplicates.txt
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

python ../filter_spirs_sarc.py \
--data_dir "../data/spirs_history_timestamps" \

