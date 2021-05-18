import sys
import os
import json 
from time import sleep
import tweepy
from utils import *
import numpy as np
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError

def save_numpy(data, file_name='../data/relationships.npy'):
    with open(file_name, 'wb') as f:
        np.save(f, data)

# Read users
api = init_tweet_api()
  
users = np.array(read_users())
users_info = read_json('../data/users_info.json')

relationships = np.zeros((len(users), len(users)),dtype=np.bool)

for i, user in enumerate(users):
    if i % 10 == 0:
        print('{}th user, with id {}.\n'.format(i, user), flush=True, end='')
    try:
        followed_by = api.friends_ids(user_id=user)
    except Exception as e:
        save_numpy(relationships)
        print('Checkpoint at {}th user'.format(i))
        print(e)
        print('Restoring connection...')
        api = init_tweet_api()
        
    indices = np.in1d(users, followed_by, assume_unique=True).nonzero()[0]
    relationships[i, indices] = 1
    


save_numpy(relationships)
