import re
import string
import pandas as pd
import os 
import torch
import json 
import tweepy
from credentials import *
import numpy as np
import pylab as plt
import networkx as nx
import torch.nn.functional as F
from constants import *
import re
import emoji
from tqdm import tqdm
from .metrics import *


def print_metrics(cm):
        print(cm)

        metrics = MultiClassMetrics()

        metrics.update(cm)
        num_labels = len(cm)

        if num_labels == 3:
            idtolabels = ['non_sar', 'INTENDED', 'PERCEIVED']
        else:
            idtolabels = ['nonsar', 'sar', 'cue', 'obl', 'eli', 'user']
            
        print(('{} \t {} \t {} \t {}').format('label', 'precision', 'recall', 'f1 score'))
        for i in range(num_labels):
            metric = metrics.class_metrics[i]
            print(('{} \t {:2f} \t {:2f} \t {:2f}').format(idtolabels[i], metric.precision, metric.recall, metric.f1_score))


def fill_zeros_with_random(user_embeddings, bottom=-1, top=1):
    temp = torch.zeros(user_embeddings.size()[1]).double()
    
    for i, user in enumerate(user_embeddings):
        user = user.cpu()

        if torch.equal(temp, user):
            user_embeddings[i] = torch.FloatTensor(user.size()).uniform_(bottom, top)
            
    return user_embeddings


def normalize(input):
    input_mean = torch.mean(input)
    input_std = torch.std(input)
    input = input - input_mean
    input = input / input_std
    return input


def compute_density(full_graph, labels):
    degrees = list(full_graph.degree())
    degrees = [d for i, d in degrees]

    n_nodes = full_graph.number_of_nodes() - degrees.count(0)
    total_edges = sum(degrees)
    max_edges = n_nodes * (n_nodes - 1)

    density = total_edges / max_edges
    return density


def compute_homophily(full_graph, tweet_start, labels):
    """Computes homophily of the users in the graph

    Args:
        full_graph (nx.Graph()): Social network created from both users and tweets
        tweet_start (int): index of tweet start. (Can be removed)
        labels (list): list of labels for each node in the graphs 

    Returns:
        homophily [float32]: Homophily amount of users in the graph. 
    """    
    degrees = list(full_graph.degree())
    degrees = [d for i, d in degrees]

    count = 0
    
    for node in tqdm(range(tweet_start)):
        source_label = None
        both_labels = False


        for n in full_graph.neighbors(node):
            #if labels[n] == 1 or labels[n] == 0:
            if labels[n] != 5:
                if source_label is None:
                    source_label = labels[n] 
                else: 
                    both_labels = True

        for n in full_graph.neighbors(node):
            if labels[n] == 5:
                target_label = None
                target_bothLabels = False

                for t in full_graph.neighbors(n):
                    #if labels[t] == 1 or labels[t] == 0:
                    if labels[t] != 5:
                        if target_label is None:
                            target_label = labels[t] 
                        else: 
                            target_bothLabels = True

                if (target_label == source_label) or both_labels or target_bothLabels:
                    count += 1

    homophily = count / len(degrees) # TODO: needs to be -> full_graph.number_of_edges()
    return homophily


def fill_zeros_with_random(user_embeddings, bottom=-1, top=1):
    temp = torch.zeros(user_embeddings.size()[1]).double()
    c = 0
    
    for i, user in enumerate(user_embeddings):
        user = user.cpu()

        if torch.equal(temp, user):
            c += 1
            user_embeddings[i] = torch.FloatTensor(user.size()).uniform_(bottom, top)
    
    print("Filled {} entries".format(c))
    return user_embeddings


def get_src_trg(full_graph):
    source = []
    target = []
    for s, t in full_graph.edges:
        source.append(s)
        target.append(t)

    assert len(source) == len(target)
    
    return source, target

def graph_analize(graph, name):
    degrees = sorted(d for n, d in graph.degree())
    print("Analyzing {} graph".format(name))
    print("Number of all nodes for {} graph: {}".format(name, len(graph.nodes())))
    print("Number of all edges for {} graph: {}".format(name, len(graph.edges())))
    print("Number of nodes with degree equal to zero: {}".format(sum([1 for x in degrees if x == 0])))
    print("Number of nodes with degree equal to one: {}".format(sum([1 for x in degrees if x == 1])))
    print("Number of nodes with degree greater than zero: {}".format(sum([1 for x in degrees if x  != 0])))

def draw_graph(temp):
    plt.figure(1,figsize=(14,14))
    nx.draw(temp, with_labels=True, node_size=8000,font_size=22)
    plt.savefig("graph.png")


def get_embeddings_dict_from_path(path):
    if not os.path.exists(path):
        print("{} does not exist !".format(path))
        return

    embeddings = {}

    with open(os.path.join(path), 'r') as f:
        for i, line in enumerate(f.readlines()):
            temp = line.strip().split(' ')
            user = temp[0]
            embeddings[user] = np.array(temp[1:], dtype=np.double)
    
    return embeddings


def write_user_embeddings(path, user_embeddings):
    """ Write user embeddings in a text file given by {path}.
    Args:
        path: (string) text file path 
        user_embeddings: (dict) contains users as keys and their learned embeddings as values
    """
    with open(path, 'w') as f:
        for user, embedding in user_embeddings.items():
                temp = str(user)

                for val in embedding:
                    temp += ' ' + str(round(val.item(), 4))

                f.write(temp + '\n')


def save_checkpoint(state, checkpoint, name='last.pth.tar'):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'.
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, name)
    if not os.path.exists(checkpoint):
        #print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)

    torch.save(state, filepath)

def init_tweet_api():
    try:
        CONSUMER_KEY
        CONSUMER_SECRET
    except:
        print('Edit credentials.py to add your Twitter API credentials in the first two lines (CONSUMER_KEY and CONSUMER_SECRET)')
        print('See here for more information on getting API credentials: https://developer.twitter.com/en/apps')
        exit(1)
        
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    return api

def write_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)

def read_json(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    
    return data

def read_users(f_name='users.txt'):
        with open('users.txt', 'r') as f:
            users_ids = []
            for line in f.readlines(): 
                users_ids.append(int(line.strip()))
        return users_ids


EMOJI_DESCRIPTION_SCRUB = re.compile(r':(\S+?):')
HASHTAG_BEFORE = re.compile(r'#(\S+)')
BAD_HASHTAG_LOGIC = re.compile(r'(\S+)!!')
FIND_MENTIONS = re.compile(r'@(\S+)')
LEADING_NAMES = re.compile(r'^\s*((?:@\S+\s*)+)')
TAIL_NAMES = re.compile(r'\s*((?:@\S+\s*)+)$')

def process_tweet(s, save_text_formatting=True, keep_emoji=False, keep_usernames=False):
    if save_text_formatting:
        s = re.sub(r'https\S+', r'', str(s))
    else:
        s = re.sub(r'https\S+', r' ', str(s))
        s = re.sub(r'x{3,5}', r' ', str(s))
    
    s = re.sub(r'\\n', ' ', s)
    s = re.sub(r'\s', ' ', s)
    s = re.sub(r'<br>', ' ', s)
    s = re.sub(r'&amp;', '&', s)
    s = re.sub(r'&#039;', "'", s)
    s = re.sub(r'&gt;', '>', s)
    s = re.sub(r'&lt;', '<', s)
    s = re.sub(r'\'', "'", s)

    if save_text_formatting:
        s = emoji.demojize(s)
    elif keep_emoji:
        s = emoji.demojize(s)
        s = s.replace('face_with', '')
        s = s.replace('face_', '')
        s = s.replace('_face', '')
        s = re.sub(EMOJI_DESCRIPTION_SCRUB, r' \1 ', s)
        s = s.replace('(_', '(')
        s = s.replace('_', ' ')

    s = re.sub(r"\\x[0-9a-z]{2,3,4}", "", s)
    
    if save_text_formatting:
        s = re.sub(HASHTAG_BEFORE, r'\1!!', s)
    else:
        s = re.sub(HASHTAG_BEFORE, r'\1', s)
        s = re.sub(BAD_HASHTAG_LOGIC, r'\1', s)
    
    if save_text_formatting:
        #@TODO 
        pass
    else:
        # If removing formatting, either remove all mentions, or just the @ sign.
        if keep_usernames:
            s = ' '.join(s.split())

            s = re.sub(LEADING_NAMES, r' ', s)
            s = re.sub(TAIL_NAMES, r' ', s)

            s = re.sub(FIND_MENTIONS, r'\1', s)
        else:
            s = re.sub(FIND_MENTIONS, r' ', s)
    #s = re.sub(re.compile(r'@(\S+)'), r'@', s)
    user_regex = r".?@.+?( |$)|<@mention>"    
    s = re.sub(user_regex," @user ", s, flags=re.I)
    
    # Just in case -- remove any non-ASCII and unprintable characters, apart from whitespace  
    s = "".join(x for x in s if (x.isspace() or (31 < ord(x) < 127)))
    s = ' '.join(s.split())
    return s
