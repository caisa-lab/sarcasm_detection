#!/bin/bash
#
#SBATCH --job-name=tweet_embeddings
#SBATCH --output=/ukp-storage-1/plepi/users_sarcasm/outputs/tweet_embeddings.txt
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
python ../tweet_embeddings.py \
--graph_path '../data/graph_network_full_with_cue.pkl' \
--out_file '../data/tweet_embeddings_paraphrase.pkl' \
--bert_model 'paraphrase-distilroberta-base-v1'