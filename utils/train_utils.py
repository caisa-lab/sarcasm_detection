import torch 
from sklearn.model_selection import train_test_split
    

def get_user_idx(mask, tweet_start, graph_network, user_label):
    user_idx = []

    for i, mask in enumerate(mask):
        val = mask.item()

        if val:
            idx = tweet_start + i

            for n in graph_network.full_graph[idx]:
                if graph_network.labels[n] == user_label:
                    user_idx.append(n)
                    break
    
    return torch.tensor(user_idx)

def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()


def get_train_val_test_masks(graph_network, tweet_start, tweets_count, test_size):
    labelstoid = graph_network.labelstoid
    print(labelstoid)

    ctr = 0
    subgraphs = []
    sub_labels = []

    temp_labels = []
    temp = []

    for idx, label in enumerate(graph_network.labels[tweet_start:]):
        if label in temp_labels and label != labelstoid['user']:
            subgraphs.append(temp)
            sub_labels.append(temp_labels)
            temp = []
            temp_labels = []
        
        nodeId = idx + tweet_start
        temp.append(nodeId)
        temp_labels.append(label)

        ctr += 1


    train, test = train_test_split(subgraphs, test_size=test_size, random_state=11)
    train, val = train_test_split(train, test_size=0.11, random_state=11)


    train_mask = torch.zeros(tweets_count, dtype=torch.bool)
    val_mask = torch.zeros(tweets_count, dtype=torch.bool)
    test_mask = torch.zeros(tweets_count, dtype=torch.bool)

    for sub in train:
        for idx in sub:
            train_mask[idx-tweet_start] = True

    for sub in val:
        for idx in sub:
            val_mask[idx-tweet_start] = True

    for sub in test:
        for idx in sub:
            test_mask[idx-tweet_start] = True and (not train_mask[idx-tweet_start])

    print('Tweet counts in train, val and test splits: {}, {}, {}'.format(train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item()))

    return train_mask, val_mask, test_mask