import argparse
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser(description='User history stats')

parser.add_argument('--user_history_path', help='user history file path')
parser.add_argument('--out_file', help='out file')

args = parser.parse_args()

user_tweetscount = {}
with open(args.user_history_path, 'r+') as f:
    
    for line in tqdm(f.readlines()):
        user_id, tweet_id, tweet_text, created_at = line.split('\t')
        user_tweetscount[user_id] = user_tweetscount.get(user_id, 0) + 1

pickle.dump(user_tweetscount, open(args.out_file, 'wb'))