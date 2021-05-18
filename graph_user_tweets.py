import torch

import numpy as np
import json 
import networkx as nx

class GraphUserTweets:
    def __init__(self, dataframe, users_included, col_names):
        self.idtousername = {} # user id to their username, not used atm 
        self.labels = [] # labels 
        self.labelstoid = {'non_sar': 0, 'sar': 1, 'cue': 2, 'obl': 3, 'eli': 4, 'user': 5}
        self.sarTypetoid = {'non_sar': 0, 'INTENDED': 1, 'PERCEIVED': 2}
        self.sarTypes = []
        self.tweet_mask = []
        self.tweet_representation = {}
        self.hashtotweettext = {}

        self.idtoindex = {} # user id or tweet id to node index
        self.indextoid = [] # node index to user id or tweet id

        self.comment_graph = None
        self.mention_graph = None
        self.full_graph = None
        self.dataframe = dataframe
        self.users_included = users_included
        self.col_names = col_names

    def add_user(self, user_id, username):
        user_id = str(user_id)
        if user_id not in self.idtoindex:
            self.idtousername[user_id] = username
            self.idtoindex[user_id] = len(self.idtoindex)
            self.indextoid.append(user_id)
            self.labels.append(self.labelstoid['user'])
            self.sarTypes.append(0)
            self.tweet_mask.append(0)
            assert self.indextoid[self.idtoindex[user_id]] == user_id 
    
    def add_tweet(self, hash_tweet, tweet, tweeType, sarType):
        if hash_tweet not in self.idtoindex:
            self.hashtotweettext[hash_tweet] = tweet
            self.idtoindex[hash_tweet] = len(self.idtoindex)
            self.indextoid.append(hash_tweet)
            self.labels.append(self.labelstoid[tweeType])
            self.sarTypes.append(self.sarTypetoid[sarType])
            self.tweet_mask.append(1)
            assert self.indextoid[self.idtoindex[hash_tweet]] == hash_tweet 

    def get_idx(self, id):
        if id in self.idtoindex:
            return self.idtoindex[id]
        return None
    
    def get_user_embeddings(self, users_vocab, path, dim=400):
        user_embeddings = None
        users_count = 0
        with open(path, 'r') as f:
            temp_emb = np.zeros((len(users_vocab.itos), int(dim)))
            
            for line in f.readlines():
                temp = line.strip().split(' ')
                user = temp[0]
                
                if user in users_vocab.stoi and len(temp) > 1:
                    idx = users_vocab.stoi[user]
                    temp_emb[idx] = np.array(temp[1:], dtype=np.double)
                    users_count += 1

            user_embeddings = torch.nn.Embedding.from_pretrained(torch.tensor(temp_emb))
        
        print("Users found / Total -> {} / {}".format(users_count, len(users_vocab.itos)))
            
        return user_embeddings

    def get_comment_graph(self):
        if self.comment_graph is not None:
            return self.comment_graph
        
        user_col_names = self.col_names['user']
        self.comment_graph = nx.Graph()
        
        for row in self.dataframe[user_col_names].iterrows():
            templist = list()
            for user in row[1]:
                if type(user) == str and '|' in user:
                    user_tokens = user.split('|')
                    user_id = int(user_tokens[-1])
                    username = '|'.join(user_tokens[:-1])
                    
                    if user_id not in templist:
                        templist.append(user_id)
                        self.add_user(user_id, username)
                        self.comment_graph.add_node(self.get_idx(str(user_id)))
            
            for i, user in enumerate(templist):
                if i < len(templist) - 1:
                    for target in templist[i+1:]:
                        if str(user) != str(target):
                            self.comment_graph.add_edge(self.get_idx(str(user)), self.get_idx(str(target)))
        
        return self.comment_graph
    

    def add_tweet_nodes_full_graph(self):
        user_col_names = self.col_names['user']
        tweet_col_names = self.col_names['tweet']
        tweet_id_col_names = self.col_names['tweet_id']
        cols = self.col_names['cols']


        for row in self.dataframe.iterrows():
            values = row[1]

            label = values['label']
            idx = values['idx']
            users = values[user_col_names]
            tweets = values[tweet_col_names]
            tweet_ids = values[tweet_id_col_names]
            templist = list()
            
            for i, (user, tweet, tweet_id) in enumerate(zip(users, tweets, tweet_ids)):
                if tweet != '':
                    tweetType = cols[i]
                    tweet_hash = tweet_id
                    sarType = 'non_sar'

                    if label == 0 and tweetType == 'sar':
                        tweetType = 'non_sar'

                    if tweetType == 'sar':
                        sarType =  values['perspective']

                    user_tokens = user.split('|')
                    user_id = int(user_tokens[-1])
                    username = '|'.join(user_tokens[:-1])

                    if tweet_hash not in templist:
                        templist.append(tweet_hash)

                    self.add_tweet(tweet_hash, tweet, tweetType, sarType)

                    if self.users_included:
                        self.full_graph.add_edge(self.get_idx(str(user_id)), self.get_idx(tweet_hash))


            for i, tweet_hash in enumerate(templist):
                if i < len(templist) - 1:
                    for target in templist[i+1:]:
                        if tweet_hash != target:
                            self.full_graph.add_edge(self.get_idx(tweet_hash), self.get_idx(target))
      

    def get_mention_graph(self, mention_path):
        if self.mention_graph is not None:
            return self.mention_graph

        self.mention_graph = nx.Graph()

        users_mentions = json.load(open(mention_path, 'r+'))

        for user, tweet_mentions in users_mentions.items():
            user = np.int64(user)
            self.add_user(user, "")

            self.mention_graph.add_node(self.get_idx(str(user)))
            for temp in tweet_mentions:
                for tweet_mention in temp:
                    tweet, mention = tweet_mention
                    self.add_user(mention, "")
                    self.mention_graph.add_node(self.get_idx(str(mention)))
                    
                    if  str(user) != str(mention):
                        self.mention_graph.add_edge(self.get_idx(str(user)), self.get_idx(str(mention)), tweet=tweet)
        
        return self.mention_graph

    def get_full_graph(self, mention_path, ):
        if self.users_included:
            print("Creating graph with users included !")
            self.mention_graph = self.get_mention_graph(mention_path)
            self.comment_graph = self.get_comment_graph()
            self.full_graph = nx.compose(self.mention_graph, self.comment_graph)
        else:
            print("Creating graph WITHOUT users !")
            self.full_graph = nx.Graph()

        self.add_tweet_nodes_full_graph()
        return self.full_graph