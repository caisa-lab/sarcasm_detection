source PATH/TO/conda.sh
conda activate sarcasm_detection
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH/TO/anaconda3/lib


python ../gat_train.py \
--model_dir='../data/gatClassify/' \
--data_dir='../data/' \
--config_dir='../configs' \
--user_mentions_file='user_mentions2.json' \
--user_embeddings_path='../data/user_embeddings.txt'