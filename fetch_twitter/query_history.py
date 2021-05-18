import pandas as pd
from utils.utils import init_tweet_api
from fetch_twitter.twitter import Twitter
import tweepy
from credentials import *
import os
import re
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='filters the dataset by removing all users without a valid context.')
parser.add_argument('--in_file', help='file with all users in the dataset.')
parser.add_argument('--out_file', help='file path to write the resulting [user_id, tweet_id, text, created_at] tuples to.')

if __name__ == '__main__':
    args = parser.parse_args()
    twitter = Twitter(CONSUMER_KEY,CONSUMER_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    
    # create the file initially, if it doesn't already exist, so it can be reused as a cache over multiple runs.
    if not os.path.exists(args.out_file):
        open(args.out_file, 'x', encoding='utf-8').close()

    # read all user ids into memory (it's not really that much).
    with open(args.in_file, encoding='utf-8') as in_file:
        all_users = {line.strip() for line in in_file}

    print('{} total users'.format(len(all_users)))

    # read output file for the first time, checking which user histories have already been crawled.
    with open(args.out_file, encoding='utf-8') as out_file:
        user_cache = {line.split('\t')[0] for line in out_file}
        
    print('{} users in cache, continue to query...'.format(len(user_cache)))
        
    # open output file again, this time in append mode, to continue crawling user histories.
    with open(args.out_file, 'a') as out_file:
        for user_id in tqdm(all_users):
            if user_id not in user_cache:
                try:
                    # first, cache all tweets of a given user. This way, if the script crashes while querying a
                    # user, it won't be in the file yet and be requeried the next time.
                    # Perform minimal preprocessing, so everything fits in one line and spaces are normalized.
                    cache = []
                    for tweet in twitter.lookup_user(user_id):
                        tweet_id = getattr(tweet, 'id')
                        tweet_text = re.sub(r'\s+', ' ', getattr(tweet, 'full_text', '').replace('\n', ' '))
                        created_at = getattr(tweet, 'created_at')
                        cache.append((tweet_id, tweet_text, created_at))
                    # write the user history to the file. There should be no issue at this point.
                    for tweet_id, tweet_text, created_at in cache:
                        print('{}\t{}\t{}\t{}'.format(user_id, tweet_id, tweet_text, created_at), file=out_file)
                except tweepy.TweepError as e:
                    # user profile not visible
                    if e.api_code == 401 or e.response.status_code == 401:
                        print('encountered HTTP 401 for user {}'.format(user_id))
                    # user profile deleted
                    elif e.api_code == 404 or e.response.status_code == 404:
                        print('encountered HTTP 404 for user {}'.format(user_id))
                    # something else happened, resolve manually. Usually, this is some form of internal server
                    # error (HTTP 5xx) and the script just needs to be restarted.
                    else:
                        raise e
                finally:
                    user_cache.add(user_id)
    print('finished reading {} / {} users'.format(len(user_cache), len(all_users)))