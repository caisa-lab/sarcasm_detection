import torch
from torch import nn

from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from torch_geometric.data import DataLoader as GraphLoader

from sentence_transformers import SentenceTransformer, models

import os
import sys
import json
import sys
import argparse
import pickle
from tqdm import tqdm 
from enum import Enum
import logging

from spirs_dataset import Spirs
from graph_user_tweets import GraphUserTweets
from constants import *
from utils.train_utils import *
from utils.utils import *
from utils.graph_utils import *
from utils.metrics import *
from utils.loss_fct import * 
from models.gat_model import GatModel

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

from sklearn.utils import shuffle

logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Train bert-gat model')

parser.add_argument('--model_dir', help='model directory')
parser.add_argument('--data_dir', help='data directory')
parser.add_argument('--config_dir', help='configs directory')
parser.add_argument('--user_mentions_file', help='user mentions file insisde data directory')
parser.add_argument('--user_embeddings_path', help='user embeddings path')
parser.add_argument('--seed', default=1234)


def train(model, data, optimizer, labels, tweets_mask, data_mask, user_idx, samples_per_cls, gat_configs):
    num_labels = gat_configs['num_labels']
    model.train()
    
    logits = model(data.to(DEVICE), tweets_mask.to(DEVICE), data_mask.to(DEVICE), user_idx, False)
    loss = loss_fn(logits.view(-1, num_labels), labels.view(-1)[data_mask.to(DEVICE)], samples_per_cls, num_labels)
            
    optimizer.zero_grad()
    loss.backward()
    #torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()

    return loss


def evaluate(model, data, labels, tweets_mask, data_mask, user_idx, samples_per_cls, gat_configs):
    num_labels = gat_configs['num_labels']
    model.eval()

    with torch.no_grad():
        logits, att = model(data.to(DEVICE), tweets_mask.to(DEVICE), data_mask.to(DEVICE), user_idx, True)        
        loss = loss_fn(logits.view(-1, num_labels), labels.view(-1)[data_mask.to(DEVICE)], samples_per_cls, num_labels)
    
    return loss, logits, att


def loss_fn(output, targets, samples_per_cls, no_of_classes=2):
    beta = 0.9999
    gamma = 2.0
    loss_type = "softmax"

    if loss_type == "softmax":
        return nn.CrossEntropyLoss()(output, targets)

    return CB_loss(targets, output, samples_per_cls, no_of_classes, loss_type, beta, gamma)

if __name__== "__main__":
    args = parser.parse_args()
    
    path = os.path.abspath(os.getcwd())
    sys.path.append(os.path.dirname(path)) # parent directory

    torch.manual_seed(args.seed)
    config_dir = args.config_dir
    gat_configs = json.load(open(os.path.join(config_dir, "gat_config.json"), "r"))
    dim = gat_configs['in_dim']
    logging.info('#' * 80)
    logging.info('Configurations: {}'.format(gat_configs))

    data_dir = args.data_dir
    sarcastic_path = os.path.join(data_dir, 'SPIRS-sarcastic.csv')
    nonsarcastic_path = os.path.join(data_dir, 'SPIRS-non-sarcastic.csv')

    spirs = Spirs()
    dataframe = spirs.read_dataset(sarcastic_path, nonsarcastic_path)
    dataframe = shuffle(dataframe, random_state=111).reset_index(drop=True)


    users_included = bool(gat_configs["users_included"])
    logging.info("Users included: {}".format(users_included))
    graphType = GraphTypes.FULL
    colNames = GraphTypeToColNames.get_col_names(graphType)
    logging.info("Graph type {}".format(graphType))
    logging.info('Column names', colNames)
    graph_path = os.path.join(data_dir, f'graph_network_{graphType.value}.pkl')

    if os.path.exists(graph_path):
        logging.info("Loading graph")
        graph_network = pickle.load(open(graph_path, 'rb'))
    else:
        logging.info("Creating new social graph !")
        mention_path = os.path.join(data_dir, args.user_mentions_file)
        graph_network = GraphUserTweets(dataframe, users_included)
        full_graph = graph_network.get_full_graph(mention_path)
        pickle.dump(graph_network, open(graph_path, 'wb'))

    tweet_start = graph_network.tweet_mask.index(1)
    logging.info('Number of nodes in the graph: {}'.format(len(graph_network.indextoid)))
    indextouser = graph_network.indextoid[:tweet_start]
    usertoindex = {}

    for idx, user in enumerate(indextouser):
        usertoindex[user] = idx

    users_vocab_path = os.path.join(data_dir, f'users_vocab_{graphType.value}.pkl')

    if os.path.exists(users_vocab_path): 
        logging.info("Loading user vocab")
        users_vocab =  pickle.load(open(users_vocab_path, 'rb'))
    else:
        logging.info("Creating user vocab !")
        dataset = spirs.build_dataset(dataframe)
        spirs.users_field.vocab.itos = indextouser
        spirs.users_field.vocab.stoi = usertoindex
        users_vocab = spirs.users_field.vocab
        pickle.dump(users_vocab, open(users_vocab_path, 'wb'))


    if users_included:
        user_embeddings_path = args.user_embeddings_path
        if not os.path.exists(user_embeddings_path):
            logging.error("{} not found".format(user_embeddings_path))
            sys.exit(0)

        user_embeddings = graph_network.get_user_embeddings(users_vocab, user_embeddings_path, dim=dim)
        user_embeddings = user_embeddings.weight.to(DEVICE)
        user_embeddings = fill_zeros_with_random(user_embeddings)

        logging.info('Size of user embeddings: {}'.format(user_embeddings.size()))

    tweets_count = len(graph_network.indextoid[tweet_start:])

    logging.info('Number of tweets in the graph: {}'.format(tweets_count))

    tweet_embeddings = pickle.load(open(os.path.join(data_dir, f'tweet_embeddings_{dim}.pkl'), 'rb'))

    embeddings = []

    for tweet in graph_network.indextoid[tweet_start:]:
        embeddings.append(tweet_embeddings[tweet])

    tweet_embeddings = torch.stack(embeddings).to(DEVICE)

    if users_included:
        representations = torch.cat([user_embeddings, tweet_embeddings])
    else:
        representations = tweet_embeddings

    logging.info('Size of all node representations: {}'.format(representations.size()))
    
    # Init GAT
    source, target = get_src_trg(graph_network.full_graph)

    num_labels = gat_configs["num_labels"]
    model = GatModel(gat_configs).to(DEVICE)
    logging.info(model)
    
    x = representations
    y = torch.zeros(x.size()[0])
    data = Data(x=x.double(), y=y, edge_index=torch.tensor([source, target], dtype=torch.long)).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=gat_configs["lr"])
    model_dir = args.model_dir
    epochs = gat_configs['epochs']
    best_loss = 50000

    if num_labels == 2:
        labels = torch.tensor([l if l == 1 else 0 for l in graph_network.labels[tweet_start:]]).to(DEVICE)
    elif num_labels == 3:
        labels = torch.tensor([l for l in graph_network.sarTypes[tweet_start:]]).to(DEVICE)
    else:
        labels = torch.tensor([l for l in graph_network.labels[tweet_start:]]).to(DEVICE)
    
    test_size = gat_configs["test_size"]
    logging.info("Test size is {}".format(test_size))
    train_mask, val_mask, test_mask = get_train_val_test_masks(graph_network, tweet_start, tweets_count, test_size)
    tweets_mask = torch.tensor(graph_network.tweet_mask, dtype=torch.bool).view(-1, 1).to(DEVICE)
    logging.info('Size of tweets mask: {}'.format(tweets_mask.size()))
    logging.info('Size of train mask: {}'.format(train_mask.sum()))
    logging.info('Size of graph network tweet mask {}'.format(len(graph_network.tweet_mask)))
    
    train_labels = labels.view(-1)[train_mask.to(DEVICE)].cpu()
    samples_per_cls_train = get_samples_per_class(train_labels)
    val_labels = labels.view(-1)[val_mask.to(DEVICE)].cpu()
    samples_per_cls_val = get_samples_per_class(val_labels)
    test_labels = labels.view(-1)[test_mask.to(DEVICE)].cpu()
    samples_per_cls_test = get_samples_per_class(test_labels)

    if num_labels < 4:
        logging.info("Filtering only sarcastic and non sarcastic tweets.")
        sarcastic_mask = torch.tensor([True if (l == 1 or l ==  0) else False for l in graph_network.labels[tweet_start:]], dtype=torch.bool)
        train_mask = torch.logical_and(sarcastic_mask, train_mask)
        val_mask = torch.logical_and(sarcastic_mask, val_mask)
        test_mask = torch.logical_and(sarcastic_mask, test_mask)
        
        logging.info("After filtering size of train set {}, val set {}, test set {}".format(train_mask.sum(), val_mask.sum(), test_mask.sum()))
        if num_labels == 3:
            logging.info("Cleaning non sarcastic tweets.")
            def remove_nonsarc(mask, labels):
                
                for i, _ in enumerate(mask):
                    if labels[i] == 0:
                        mask[i] = 0
                
            remove_nonsarc(train_mask, labels)
            remove_nonsarc(test_mask, labels)
            remove_nonsarc(val_mask, labels) 

            labels = torch.tensor([l - 1 for l in labels]).to(DEVICE)
            gat_configs["num_labels"] = 2
            num_labels = 2
            samples_per_cls_train = samples_per_cls_train[1:]
            samples_per_cls_val = samples_per_cls_val[1:]
            samples_per_cls_test = samples_per_cls_test[1:] 


    # create user idx masks
    user_label = graph_network.labelstoid['user']
    train_user_idx = get_user_idx(train_mask, tweet_start, graph_network, user_label).to(DEVICE)
    val_user_idx = get_user_idx(val_mask, tweet_start, graph_network, user_label).to(DEVICE)
    test_user_idx = get_user_idx(test_mask, tweet_start, graph_network, user_label).to(DEVICE)


    for epoch in range(epochs):
        
        train_loss =  train(model, data, optimizer, labels, tweets_mask, train_mask, train_user_idx, samples_per_cls_train, gat_configs)
        val_loss, _, _ = evaluate(model, data, labels, tweets_mask, val_mask, val_user_idx, samples_per_cls_val, gat_configs)
        
        if val_loss.item() < best_loss:
            best_loss = val_loss.item()

            save_checkpoint({'epoch': epoch + 1,
                                                'state_dict': model.state_dict(),
                                                'optim_dict': optimizer.state_dict()},
                                                checkpoint=model_dir, name=f'best_model.tar')

        if epoch % 100 == 0:
            print("Epoch {} \t | \t Train Loss: {:.4f} \t | \t Validation Loss: {:.4f}".format(epoch+1, train_loss.item(), val_loss.item()))


    model_path = os.path.join(model_dir, 'best_model.tar')
    checkpoint = torch.load(model_path, map_location=DEVICE)
    logging.info("Checkpoint was in epoch {}".format(checkpoint['epoch']))
    model.load_state_dict(checkpoint['state_dict'])

    loss, logits, att = evaluate(model, data, labels, tweets_mask, test_mask, test_user_idx, samples_per_cls_test, gat_configs)
    predictions = logits.view(-1, num_labels).cpu().max(dim=1)[1]
    gold = labels.view(-1)[test_mask.to(DEVICE)].cpu()
    test_predictions = predictions
    test_gold = gold

    f1Score = f1_score(gold, predictions, average='macro')
    logging.info("Total f1 score macro {:3f}: ".format(f1Score))
    f1Score = f1_score(gold, predictions, average='micro')
    logging.info("Total f1 score micro {:3f}:".format(f1Score))
    accuracy = accuracy_score(gold, predictions)
    logging.info("Accuracy {:3f}:".format(accuracy))

    logging.info("#" * 80)
    cm_test = np.array(confusion_matrix(gold, predictions))
    logging.info("The loss in the test set: {}".format(loss.item()))
    logging.info("Evaluation in test set!")
    print_metrics(cm_test)

    logging.info("\n")
    logging.info("-" * 80)
    logging.info("\n")

    loss, logits, att = evaluate(model, data, labels, tweets_mask, train_mask, train_user_idx, samples_per_cls_train, gat_configs)
    predictions = logits.view(-1, num_labels).cpu().max(dim=1)[1]
    gold = labels.view(-1)[train_mask.to(DEVICE)].cpu()
    cm_train = np.array(confusion_matrix(gold, predictions))
    logging.info("The loss in the train set: {}".format(loss.item()))
    logging.info("Evaluation in train set! ")
    print_metrics(cm_train)

    
    