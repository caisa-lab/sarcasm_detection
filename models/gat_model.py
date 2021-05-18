import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GatModel(nn.Module):
    def __init__(self, gat_configs):
        super(GatModel, self).__init__() 
        self.in_dim = gat_configs["in_dim"]
        in_dim = self.in_dim
        out_dim = gat_configs["out_dim"] 
        heads = gat_configs["heads"]
        concat = bool(gat_configs["concat"])
        self.linear_dim = in_dim
        self.users_included = bool(gat_configs["users_included"])

        if not concat:
            self.linear_dim = out_dim
        if self.users_included:
            self.linear_dim = 2 * self.linear_dim 
        
        # out_dim, out_dim * heads, out_dim * heads
        
        self.gat1 = GATConv(in_dim, in_dim // 2, heads=heads, dropout=gat_configs["dropout"], concat=concat)
        self.gat1.double()

        self.gat2 = GATConv(in_dim // 2, in_dim // 4, heads=heads, dropout=gat_configs["dropout"], concat=concat)
        self.gat2.double()

        self.gat3 = GATConv(in_dim // 4, out_dim, heads=heads, dropout=gat_configs["dropout"], concat=concat)
        self.gat3.double()

        self.dropout = nn.Dropout(p=gat_configs["dropout"])
        self.elu = nn.ELU()
        self.relu = nn.ReLU()

        self.scale_down = nn.Linear(self.linear_dim, self.linear_dim // 2)
        self.classify = nn.Linear(self.linear_dim // 2, gat_configs["num_labels"])


    def forward(self, data, tweets_mask, data_mask, user_idx=None, return_attention_weights=None):
        if return_attention_weights:
            output,att1 = self.gat1(data.x, data.edge_index,  return_attention_weights=return_attention_weights)
            output = self.elu(output)

            output, att2 = self.gat2(output, data.edge_index,  return_attention_weights=return_attention_weights)
            output = self.elu(output)

            output, att3 = self.gat3(output, data.edge_index, return_attention_weights=return_attention_weights)
            output = self.elu(output)
        else:
            output = self.gat1(data.x, data.edge_index)
            output = self.elu(output)

            output = self.gat2(output, data.edge_index)
            output = self.elu(output)

            output = self.gat3(output, data.edge_index)
            output = self.elu(output)
      
        tweet_embeddings = output.masked_select(tweets_mask).view(-1, output.size()[1]).float() 
        tweet_embeddings = tweet_embeddings[data_mask]
        embeddings = tweet_embeddings  

        if self.users_included:
            user_embeddings = output[user_idx]
            embeddings = torch.cat([tweet_embeddings, user_embeddings], dim=1)

        embeddings = self.dropout(embeddings)
        outputs = self.scale_down(embeddings.float())
        
        outputs = self.dropout(self.relu(outputs))
        
        logits = self.classify(outputs)

        if return_attention_weights:
            return logits, (att1, att2, att3)
        
        return logits