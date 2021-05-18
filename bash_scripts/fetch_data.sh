source PATH/TO/conda.sh
conda activate sarcasm_detection
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:PATH/TO/anaconda3/lib

echo "Fetching tweets from spirs dataset."
python ../fetch_twitter/fetch-tweets.py \
--data_dir='../data/'

echo "Creating user mention file."
python ../fetch_twitter/fetch_user_mentions.py \
--data_dir='../data/'

echo "Create users file."
python ../users/get_users.py \
--data_dir='../data/'

echo "Query users history."
python ../fetch_twitter/query_history.py \
--in_file='../data/users.txt' \
--out_file='../data/full_history.txt'