import argparse
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, models
from constants import *
import torch.nn as nn
import numpy as np
from utils import process_tweet

parser = argparse.ArgumentParser(description='User history stats')

parser.add_argument('--user_history_path', help='user history file path')
parser.add_argument('--out_file', help='out file')
parser.add_argument('--bert_model', help='Bert model')

args = parser.parse_args()
torch.set_printoptions(precision=4)


def compute_user_representation(model, tweets, beta=1e-3, eps=1e-2):
    embedding = None
    output = torch.tensor(model.encode(tweets))
    
    for i, row in enumerate(output):
        temp = torch.clamp(row, min=0)
       
        #current = row + eps * temp * np.exp(-beta * i * 100)
        current = row 

        if embedding is None:
            embedding = current
        else:
            embedding += current

    return (1 / len(tweets)) * embedding


user_history_path = args.user_history_path
out_file = args.out_file
bert_model = args.bert_model
print("Loading user history {} \t | \t Bert model being used {} \t | \t Output path {} \t".format(user_history_path, bert_model, out_file))


#word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length=256)
#pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
#dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=400, activation_function=nn.Tanh())
#model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model]).to(DEVICE)
#model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens').to(DEVICE)
model = SentenceTransformer(bert_model).to(DEVICE)

print("Input file {}".format(user_history_path))
print("Output file {}".format(out_file))

user_embeddings = {}
tweet_cache = []
current_user = None
wrote = []

with open(user_history_path, 'r+') as f:

    for line in tqdm(f.readlines()):
        line = line.strip()
        user_id, tweet_id, tweet_text, created_at = line.split('\t')
        
        if user_id != current_user and len(tweet_cache) > 0:
            wrote.append(current_user) 
            tweet_cache.reverse()
            embedding = compute_user_representation(model, tweet_cache)
            
            user_embeddings[current_user] = embedding
            
            tweet_cache.clear()

        current_user = user_id
        cleaned_tweet = process_tweet(tweet_text)
        tweet_cache.append(cleaned_tweet)


print(len(wrote))
print(len(user_embeddings))

with open(out_file, 'w') as f:
    for user, embedding in user_embeddings.items():
        temp = str(user)
       
        for val in embedding:
            temp += ' ' + str(round(val.item(), 4))
        
        f.write(temp + '\n')
