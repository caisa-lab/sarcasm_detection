import argparse
import tweepy, csv
from utils.utils import *
from tqdm import tqdm
import re
import os
import json


parser = argparse.ArgumentParser(description='filters the dataset by removing all users without a valid context.')
parser.add_argument('--in_file', help='tab-separated file consisting of [user_id, tweet_id, tweet] tuples.')
parser.add_argument('--out_file', help='tab-separated file consisting of [tweet_id, user_id, label, text] tuples.')
parser.add_argument('--start_line', type=int, default=0, help='line where the execution was interrupted last time.')
parser.add_argument('--user_mentions_file', default='data/user_tweet_metions.json', help='tab-separated file consisting of [tweet_id, user_id, label, text] tuples.')


def read_history_files(in_path, out_path, tweet_api, user_mentions_file, start_line, user_tweet_mentions = {}):
    data_buffer = list()
    id_list = list()
    print("Starting at line {}".format(start_line))
    with open(in_path, 'r+') as infile:
        with open(out_path, 'a+') as outfile:
            try: 
                for i, line in tqdm(enumerate(infile.readlines())):
                    if i >= start_line:
                        user_id, tweet_id, tweet = line.split('\t')
                        id_list.append(tweet_id)
                        if len(id_list) >= 100:
                            response = tweet_api.statuses_lookup(id_list)
                            id_list.clear()

                            for reply in response:
                                tweet_id = reply.id
                                user_id = reply.user.id
                                created = reply.created_at
                                full_text = re.sub(r'\s+', ' ', getattr(reply, 'text', '').replace('\n', ' '))
                                user_mentions = []
                                for user_mention in reply.entities['user_mentions']:
                                    user_mentions.append((tweet_id, user_mention['id']))

                                if user_id not in user_tweet_mentions:
                                    user_tweet_mentions[user_id] = []
                                
                                if len(user_mentions) > 0:
                                    user_tweet_mentions[user_id].append(user_mentions)

                                data_buffer.append((user_id, tweet_id, full_text, created))

                            if len(data_buffer) >= 1000:
                                for data in data_buffer:
                                    outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' +str(data[2]) + '\t' + str(data[3]) + '\n')
                                data_buffer.clear()

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
                    print(e)
            finally:
                for data in data_buffer:
                    outfile.write(str(data[0]) + '\t' + str(data[1]) + '\t' +str(data[2]) + '\t' + str(data[3]) + '\n')
                data_buffer.clear()
                print("Had exception at user {} in line {}".format(user_id, i))
                json.dump(user_tweet_mentions, open(user_mentions_file, 'w+'))
                print("Trying to reconnect...")
                tweet_api = init_tweet_api()
                print("Reconnected...")

    return user_tweet_mentions

if __name__== '__main__':
    file =  os.path.realpath(__file__)
    os.chdir(os.path.dirname(file)) 
    
    args = parser.parse_args()
    
    tweet_api = init_tweet_api()

    temp = {}
    temp = json.load(open(args.user_mentions_file, 'r+'))
    print("Length of current user mentioning {}".format(len(temp)))
    temp = read_history_files(args.in_file, args.out_file, tweet_api, args.user_mentions_file, args.start_line, temp)

    with open(args.user_mentions_file, 'w+') as fp:
        json.dump(temp, fp)
