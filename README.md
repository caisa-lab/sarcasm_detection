# Perceived and Intended Sarcasm Detection with Graph Attention Networks

## Requirements:

Install environment "sarcasm_detection" from .yml file. 

`conda env create --file environment.yml`


## Fetching data

* Download from https://github.com/bshmueli/SPIRS both .csv with sarcastic and non-sarcastic tweet ids, and move those to `data/` folder.
* Extract tweets, users and the history from twitter: `bash fetch_data.sh`


## Compute tweet and user embeddings

* Create tweet embeddings: `bash compute_tweet_embeddings.sh`
* To create user2vec embeddings follow instructions in https://github.com/samiroid/usr2vec/edit/master/README.md, which takes as input `full_history.txt` file. 

## Run gat model

* `train_gat.sh`




 
