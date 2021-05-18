import sys
import os
import torch
import torch.nn as nn

from spirs_dataset import Spirs
from graph_user_tweets import GraphUserTweets
from constants import *
from sentence_transformers import SentenceTransformer, models
import argparse
from tqdm import tqdm
import pickle
from utils import process_tweet
from sklearn.utils import shuffle


parser = argparse.ArgumentParser(description='Tweet Embeddings')

parser.add_argument('--graph_path', help='graph path')
parser.add_argument('--out_file', help='out file')
parser.add_argument('--bert_model', help='Bert model')
parser.add_argument('--data_dir', help='Data directory')

args = parser.parse_args()


if __name__ == '__main__':
    torch.manual_seed(1234)
    graph_path = args.graph_path
    out_file = args.out_file
    bert_model = args.bert_model
    data_dir = args.data_dir

    print("Loading graph {} \t | \t Bert model being used {} \t | \t Output path {} \t".format(graph_path, bert_model, out_file))

    sarcastic_path = os.path.join(data_dir, 'SPIRS-sarcastic.csv')
    nonsarcastic_path = os.path.join(data_dir, 'SPIRS-non-sarcastic.csv')

    spirs = Spirs()
    dataframe = spirs.read_dataset(sarcastic_path, nonsarcastic_path)
    dataframe = shuffle(dataframe, random_state=111).reset_index(drop=True)

    mention_path = os.path.join(data_dir, args.user_mentions_file)
    graph_network = GraphUserTweets(dataframe, True)
    full_graph = graph_network.get_full_graph(mention_path)
    pickle.dump(graph_network, open(graph_path, 'wb'))
    tweet_start = graph_network.tweet_mask.index(1)
    

    word_embedding_model = models.Transformer(bert_model, max_seq_length=256)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=400, activation_function=nn.Tanh())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model]).to(DEVICE)
  
    #model = SentenceTransformer(bert_model).to(DEVICE)
    tweet_embeddings = {}
    ctr = 0

    for hash in tqdm(graph_network.indextoid[tweet_start:], desc="Embedding Progress"):
        tweet = graph_network.hashtotweettext[hash]
        tweet = process_tweet(tweet)
        output = model.encode(tweet)
        tweet_embeddings[hash] = torch.tensor(output)
        
    #tweet_embeddings = torch.stack(tweet_embeddings)
    #torch.save(tweet_embeddings, 'tweet_embeddings.pt')
    pickle.dump(tweet_embeddings, open(out_file, 'wb'))