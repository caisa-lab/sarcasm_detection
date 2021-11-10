# Perceived and Intended Sarcasm Detection with Graph Attention Networks

This repo contains code and instructions for the following paper: *Joan Plepi and Lucie Flek, Perceived and Intended Sarcasm Detection with Graph Attention Networks.* 

For more details on our work, please check our [paper](https://aclanthology.org/2021.findings-emnlp.408.pdf).



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

## Cite

```bash
@inproceedings{plepi-flek-2021-perceived-intended-sarcasm,
    title = "Perceived and Intended Sarcasm Detection with Graph Attention Networks",
    author = "Plepi, Joan  and
      Flek, Lucie",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.408",
    pages = "4746--4753",
    abstract = "Existing sarcasm detection systems focus on exploiting linguistic markers, context, or user-level priors. However, social studies suggest that the relationship between the author and the audience can be equally relevant for the sarcasm usage and interpretation. In this work, we propose a framework jointly leveraging (1) a user context from their historical tweets together with (2) the social information from a user{'}s neighborhood in an interaction graph, to contextualize the interpretation of the post. We distinguish between perceived and self-reported sarcasm identification. We use graph attention networks (GAT) over users and tweets in a conversation thread, combined with various dense user history representations. Apart from achieving state-of-the-art results on the recently published dataset of 19k Twitter users with 30K labeled tweets, adding 10M unlabeled tweets as context, our experiments indicate that the graph network contributes to interpreting the sarcastic intentions of the author more than to predicting the sarcasm perception by others.",
}
```

 
