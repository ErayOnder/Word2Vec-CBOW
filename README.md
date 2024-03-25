# Word2Vec Continous Bag of Words algorithm

A Pytorch implementation of Word2Vec CBOW algoritm from scratch!

## Directories

- `/data`: Contains data files used in the projects
- `/vocab`: Contains vocabulary for different tokenizers
- `/embeddings`: Contains embeddings for different tokenizers
- `/model`: Contains latest checkpoints for different tokenizers
- `/src`: Source files for the project
  - `/data_pipeline.py`: CustomDataset class for the Dataloader
  - `model.py`: CBOW model class
  - `params.py`: Hyperparameter class
  - `tokenizer.py`: Contains 3 different tokenizer classes
  - `trainer.py`: CBOW model trainer class
  - `vocab.py`: Vocabulary class
- `main.py`: Entry point for the program
- `evaluate.py`: A generic file for evaluation


## Requirements

- python==3.10.13
- numpy==1.26.4
- torch==2.1.2


## Overview

When "main.py" is executed, it reads the .txt training file and creates a word vocabulary regardless of the selected tokenization method. This is done to accelerate negative sampling process which will be mentioned later. Then, if the selected tokenization method is different from Word Tokenization, "tokenizer.py" and "vocab.py" are used to create the corresponding token vocabulary. If the vocabulary is created for the first time, it uploads a json file under vocab directory in the project folder with the corresponding name: tokenizer_name + "_vocab.json". If there is an existing vocabulary with the corresponding tokenizer, it just uploads the json file without going on with the process of recreating the vocabulary.

Next, the program creates CBOW input and output pairs in "data_pipeline.py". This file has CustomData class which is the custom dataset class that is created for the Dataloaders. This class applies sub-sampling and negative sampling operations before creating context and target pairs. The details of my sub-sampling and negative sampling implementations are explained in my report. 

Finally a Model class instance from "model.py" and an Adam optimizer is defined to be fed to Trainer class in "trainer.py". Also if there is a previous checkpoint, the model and optimizer is loaded from then on. 

## Instructions on running Word2vec CBOW

Your project environment should have the same structure of directories as above. The entry point of our program is "main.py". But before you start running "main.py", you should configure "params.py" file according to your own hyperparameter choosing. Important points on modifying "params.py" class:

- TOKENIZER ==>  !! essential !! name of the tokenizer, you should write specifically "WordTokenizer", 
"CharacterTrigrams" or "BytePairEncoding" if you want to run your algorithm with Word Tokenizer, Character Trigram Tokenizer or Byte Pair Encoding Tokenizer respectively.

- TRAIN_DATA_PATH, VAL_DATA_PATH ==>  !! essential !! needs to be changed if data is different

- VOCAB_SIZE ==> vocabulary size

- MIN_FREQ ==> minimum frequency before adding to vocabulary

- CBOW_N_WORDS ==> context size of the cbow model

- MAX_SEQ_LENGHT ==> maximum lenght of sequence for subword tokenizers such as Character Trigrams and BPE

- SUBSAMPLE_T, NEG_SAMPLES, NEG_SAMPLE_EXP, NS_ARRAY_LEN ==> hyperparameters for subsampling and negative sampling, not adviced on changing, except NEG_SAMPLES which are the number of negative samples. 

- SPECIALS ==> special tokens, not adviced on changing

- STOPWORDS ==> list of stopwords, optional to use

- BATCH_SIZE, EMBED_DIM, EMBED_MAX_NORM, NUM_EPOCH, LOSS ==> can be changed for hyperparameter tuning

- DEVICE ==> device to run on

- LOAD_CHECKPOINT ==> checkpoint flag

- CHECKPOINT_PATH, MODEL_PATH, EMBED_PATH, VOCAB_PATH ==> if the above directory structure is fulfilled, no need to change


## Evaluation - How to use "evaluate.py"

Evaluation file "evaluate.py" is a generic file to run evaluations on a trained model. This file loads the embeddings and vocabulary from "/embeddings". Note: Your "/embeddings" subfolder should contains these files: "WordTokenizer_embeddings.txt", "CharacterTrigrams_embeddings.txt", "BytePairEncoding_embeddings.txt", "BytePairEncoding_vocab.json"

The structure is simple, there are 3 methods you can use:
- Get accuracy on syntatic and semantic datasets
- Get similarity between two words
- Get top n similar words

How to use it:
- To evaluate the accuracy, your command should follow with TOKENIZER_NAME + "get_accuracy" + DISTANCE_MEASURE
  - Note that TOKENIZER_NAME can only be one of the three: "WordTokenizer", "CharacterTrigrams", "BytePairEncoding"
  - DISTANCE_MEASURE can either be "cos" for cosine distance or "euc" for Euclidean distance.
  - Example:  
  ```bash
  python evaluate.py WordTokenizer get_accuracy cos
  python evaluate.py BytePairEncoding get_accuracy euc


- To evaluate the top n words, your command should follow with TOKENIZER_NAME + "get_top_n" + N + WORD
  - Note that TOKENIZER_NAME can only be one of the three: "WordTokenizer", "CharacterTrigrams", "BytePairEncoding". But it is adviced to use "WordTokenizer" when using this method since other tokenizers are subword tokenizers and the returned top n words will most likely be subwords.
  - N is the top n number, needs to be an integer and WORD is the corresponding word, needs to be an string.
  - Example:  
  ```bash
  python evaluate.py WordTokenizer get_top_n 10 paris

- To evaluate the similarity metic, your command should follow with TOKENIZER_NAME + "get_similarity" + FIRST_WORD + SECOND_WORD + DISTANCE_MEASURE
  - Note that TOKENIZER_NAME can only be one of the three: "WordTokenizer", "CharacterTrigrams", "BytePairEncoding"
  - FIRST_WORD and SECOND_WORD should be strings
  - DISTANCE_MEASURE can either be "cos" for cosine distance or "euc" for Euclidean distance.
  - Example:  
  ```bash
  python evaluate.py CharacterTrigrams get_similarity fransa paris cos




