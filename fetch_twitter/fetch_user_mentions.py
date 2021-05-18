import os
from torchtext import data
import tweepy, csv
from utils.utils import *
from tqdm import tqdm
from spirs_dataset import Spirs
import json
import numpy as np
import argparse


parser = argparse.ArgumentParser(description='Create user mentioning json file.')
parser.add_argument('--data_dir', help='data directory')

if __name__ == '__main__':
    args = parser.parse_args()
    data_dir = args.data_dir
    spirs = Spirs()
    df = spirs.read_dataset(os.path.join(data_dir, 'SPIRS-sarcastic.csv'), os.path.join(data_dir, 'SPIRS-non-sarcastic.csv'))

    users_path = os.path.join(data_dir, 'user_mentions2.json')
    tweet_api = init_tweet_api()
    id_list = []
    user_tweet_mentions = {}

    if os.path.exists(users_path):
        with open(users_path, 'r+') as f:
            user_tweet_mentions = json.load(f)

    print(len(user_tweet_mentions))

    start = 0
    try:
        for row in tqdm(df.iterrows()):
            row_id = row[0]

            if row_id >= start:
                if row[1]['sar_id'] != '':
                    id_list.append(np.int64(row[1]['sar_id']))
                if row[1]['cue_id'] != '':
                    id_list.append(np.int64(row[1]['cue_id']))
                if row[1]['eli_id'] != '':
                    id_list.append(np.int64(row[1]['eli_id']))
                if row[1]['obl_id'] != '':
                    id_list.append(np.int64(row[1]['obl_id']))

                if len(id_list) >= 50:
                    response = tweet_api.statuses_lookup(id_list)
                    for reply in response:
                        tweet_id = reply.id
                        user_id = reply.user.id
                        created = reply.created_at
                        full_text = reply.text
                        user_mentions = []
                        for user_mention in reply.entities['user_mentions']:
                            user_mentions.append((tweet_id, user_mention['id']))

                        if user_id not in user_tweet_mentions:
                            user_tweet_mentions[user_id] = []
                        user_tweet_mentions[user_id].append(user_mentions)

                    id_list.clear()
    except tweepy.TweepError as e:
        print(e)
        with open(users_path, 'w+') as fp:
            json.dump(user_tweet_mentions, fp)
        print(id_list)
        print("Interrupted at row {}".format(row[0]))

    with open(users_path, 'w+') as fp:
        json.dump(user_tweet_mentions, fp)
