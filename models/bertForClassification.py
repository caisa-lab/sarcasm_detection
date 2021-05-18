import torch
from transformers import BertTokenizer, AdamW, BertModel
import torch.nn as nn


class BertForClassification(nn.Module):
    def __init__(self, bert=None, hidden_dropout_prob=0.1, hidden_size=768, num_labels=2, users_dim=200):
        super(BertForClassification, self).__init__()
        if bert is None:
            self.bert = BertModel.from_pretrained('bert-base-uncased')
        else:
            self.bert = bert
        self.dropout = nn.Dropout(hidden_dropout_prob)

        if users_dim != None:
            self.scale_down = nn.Linear(hidden_size, users_dim)
            self.classifier = nn.Linear(2 * users_dim, num_labels)
        else:
            self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self, 
        input_ids=None,
        attention_mask=None, 
        user_embeddings=None,
        ):

        outputs = self.bert(input_ids, attention_mask)
        pooled_output = outputs[1]

        
        pooled_output = self.dropout(pooled_output)
        
        if user_embeddings != None:
            pooled_output = self.scale_down(pooled_output)
            concat = torch.cat((pooled_output, user_embeddings), dim=1)
            logits = self.classifier(concat.float())
        else:
            logits = self.classifier(pooled_output)
        
        return logits, pooled_output
