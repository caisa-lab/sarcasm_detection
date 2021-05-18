from torchtext.data import Field, Dataset, Example, BucketIterator
from constants import *
from transformers import BertTokenizer
import pandas as pd
import numpy as np


class Spirs:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._init_fields()

    def _init_fields(self):
        # Not used at the moment
        self.idx_field = Field(False)
        self.pattern_field = Field(False)
        self.person_field = Field(False)
        self.perspective_field = Field(False)
        
        pad_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)
        unk_index = self.tokenizer.convert_tokens_to_ids(self.tokenizer.unk_token)

        self.text_field = Field(
                        use_vocab=False, 
                        tokenize=lambda x: x, 
                        pad_token=pad_index, 
                        unk_token=unk_index,
                        )
        self.users_field = Field(sequential=False, preprocessing=lambda x: x.split('|')[-1])
        self.label_field = Field(False)

        self.fields = [
            (IDX, self.idx_field),
            (PATTERN, self.pattern_field), 
            (PERSON, self.person_field), 
            (CUE_ID, None), 
            (SAR_ID, None), 
            (OBL_ID, None), 
            (ELI_ID, None), 
            (PERSPECTIVE, self.perspective_field),
            (CUE_TEXT, self.text_field),
            (SAR_TEXT, self.text_field),
            (OBL_TEXT, self.text_field), 
            (ELI_TEXT, self.text_field),
            (CUE_USER, self.users_field), 
            (SAR_USER, self.users_field), 
            (OBL_USER, self.users_field), 
            (ELI_USER, self.users_field),
            (LABEL, self.label_field)   
        ]
    
    def build_dataset(self, df):
        df = df[df['sar_text'] != '']
        examples = [Example.fromCSV(row[1], self.fields) for row in df.iterrows()]
        self.dataset = Dataset(examples, self.fields)

        self._build_vocabs()
        return self.dataset

    def _build_vocabs(self):
        self.specials = [PAD_TOKEN, UNK_TOKEN, START_TOKEN, END_TOKEN, SEP_TOKEN]
        self.idx_field.build_vocab(self.dataset)
        self.pattern_field.build_vocab(self.dataset)
        self.person_field.build_vocab(self.dataset)
        self.perspective_field.build_vocab(self.dataset)
        self.perspective_field.vocab.itos.reverse()
        for i, k in enumerate(self.perspective_field.vocab.itos): 
            self.perspective_field.vocab.stoi[k] = i
        
        self.label_field.build_vocab(self.dataset, specials_first=False)
        self.users_field.build_vocab(self.dataset)


    def read_dataset(self, sarcastic_path, nonsarcastic_path):
        #  dtype={'cue_id': np.int64, 'sar_id': np.int64, 'obl_id': np.int64, 'eli_id': np.int64}
        sarcastic_df = pd.read_csv(sarcastic_path, sep=None )
        sarcastic_df['label'] = [1] * len(sarcastic_df)

        nonsarcastic_df = pd.read_csv(nonsarcastic_path, sep=None)
        nonsarcastic_df['label'] = [0] * len(nonsarcastic_df)

        dataframe = pd.concat([sarcastic_df, nonsarcastic_df], axis=0, ignore_index=True)
        dataframe.fillna('', inplace=True)

        return dataframe