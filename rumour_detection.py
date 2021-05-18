# %%
import numpy as np
import json 
import networkx as nx
from ast import literal_eval

from tqdm import tqdm

import os 
import sys
# %%
data_dir = 'data/rumdetect'

twitter16_path = os.path.join(data_dir, 'twitter16')

# %%
class RumoursDataset:
    def __init__(self):
        self.graph = nx.Graph()
        self.idToNodeIdx = {}
        self.nodeIdxToId = []
        self.labelsToId = {'false': 0, 'true': 1, 'non-rumor': 2, 'unverified': 3, 'tree': 4, 'user': 5}
        self.userTweetLabels = []
        self.labels = []
        self.tweetIdToText = {}
        self.tweetIdToLabel = {}

    def read_twitter(self, twitter_path, SOURCE_TWEETS_PATH = 'source_tweets.txt', LABEL_PATH='label.txt'):
        with open(os.path.join(twitter_path, SOURCE_TWEETS_PATH)) as f:
            for line in f.readlines():
                line = line.strip()
                tweetId, tweet = line.split('\t')
                self.tweetIdToText[np.int64(tweetId)] = tweet

        with open(os.path.join(twitter_path, LABEL_PATH)) as f:
            for line in f.readlines():
                line = line.strip()
                label, tweetId = line.split(':')
                self.tweetIdToLabel[np.int64(tweetId)] = label

    def extract_values(self, array):
        return np.int64(array[0]), np.int64(array[1]), np.float(array[2])

    def add_nodes(self, nodes_dict):
        temp = []
        for k, v in nodes_dict.items():
            temp.append(k)
            if k not in self.idToNodeIdx:
                self.idToNodeIdx[k] = len(self.idToNodeIdx)
                self.nodeIdxToId.append(k)
                self.userTweetLabels.append(v)
                self.graph.add_node(self.idToNodeIdx[k])

        for s in temp[:-1]:
            for t in temp[1:]:
                self.graph.add_edge(self.idToNodeIdx[s], self.idToNodeIdx[t])

    def build_graph(self, TREE_PATH):
        for tweet in tqdm(self.tweetIdToText.keys()):
            current_path = os.path.join(TREE_PATH, str(tweet) + '.txt')
            
            if os.path.exists(current_path):
                with open(current_path, 'r') as f:
                    for line in f.readlines():
                        if 'ROOT' not in line: 
                            parent, child = line.split('->')
                            parent = np.array(literal_eval(parent))
                            child = np.array(literal_eval(child))
                            
                            p_user, p_tweet, p_time = self.extract_values(parent)
                            c_user, c_tweet, c_time = self.extract_values(child)
                            nodes = {p_user: 0, p_tweet: 1, c_user: 0, c_tweet: 1}
                            
                            self.add_nodes(nodes)

# %%
rumourDataset = RumoursDataset()
TREE_PATH = os.path.join(twitter16_path, 'tree')
# %%
rumourDataset.read_twitter(twitter16_path)
rumourDataset.build_graph(TREE_PATH)
# %%
for i, node in enumerate(rumourDataset.nodeIdxToId):
    if rumourDataset.userTweetLabels[i] == 0:
        rumourDataset.labels.append(rumourDataset.labelsToId['user'])
    elif node in rumourDataset.tweetIdToLabel:
        rumourDataset.labels.append(rumourDataset.labelsToId[rumourDataset.tweetIdToLabel[node]])
    else:
        rumourDataset.labels.append(rumourDataset.labelsToId['tree'])

# %%
rumourDataset.userTweetLabels.count(0)
# %%
