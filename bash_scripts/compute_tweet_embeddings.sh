source PATH/TO/conda.sh
conda activate sarcasm_detection
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH/TO/anaconda3/lib

python ../tweet_embeddings.py \
--graph_path '../data/graph_network_full_with_cue.pkl' \
--out_file '../data/tweet_embeddings_400.pkl' \
--bert_model 'paraphrase-distilroberta-base-v1' \
--data_dir='../data/' \
--user_mentions_file='../data/user_mentions2.json'