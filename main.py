import os
from tqdm import tqdm

import torch

from src.params import Parameters
from src.tokenizer import WordTokenizer, CharacterTrigrams, BytePairEncoding
from src.vocab import build_vocab
from src.data_pipeline import CustomData
from src.model import Model
from src.trainer import Trainer

def get_data(file_path, limit=None):
    texts = []
    with open(file_path, 'r', encoding="utf-8") as file:
        print("Reading the document: ", file_path)
        print("Maximum number of lines to read: ", limit)
        for line in tqdm(file):
            if not line.startswith("=="):
                texts.append(line.strip())
                if limit:
                    if len(texts) >= limit:
                        break
    return texts

if __name__ == "__main__":

    params = Parameters()
    print("Will run on: ", params.DEVICE)

    train_file_path = params.TRAIN_DATA_PATH
    val_file_path = params.VAL_DATA_PATH
    print("Defined tokenizer: ", params.TOKENIZER)

    print("Extracting the data from the .txt files...")
    vocab_data = get_data(train_file_path)
    assert vocab_data != None, "No data is read to be passed onto the tokenizer!"
    print("(Word) - Tokenizing the text data before creating the vocabulary...")
    word_tokenizer = WordTokenizer(params.STOPWORDS)
    word_tokens = word_tokenizer.tokenize(" ".join(vocab_data))
    print("Creating the word vocabulary...")
    word_vocab_path = os.path.join("vocab", "WordTokenizer_vocab.json")
    word_vocab = build_vocab(params, tokens=word_tokens, word_vocab_path=word_vocab_path)
    if params.TOKENIZER == "BytePairEncoding":
        print("(BPE) - Tokenizing the text data before creating the vocabulary...")
        bpe_tokenizer = BytePairEncoding(params)
        print("Creating the vocabulary for byte pair encoding...")
        main_vocab = build_vocab(params, tokenizer=bpe_tokenizer)
    elif params.TOKENIZER == "CharacterTrigrams":
        print("(CharacterTrigrams) - Tokenizing the text data before creating the vocabulary...")
        ct_tokenizer = CharacterTrigrams(params.STOPWORDS)
        if os.path.exists(params.VOCAB_PATH):
            print("Creating the vocabulary for character trigrams...")
            main_vocab = build_vocab(params,tokenizer=ct_tokenizer)
        else:
            ct_tokens = ct_tokenizer.tokenize(" ".join(vocab_data))
            print("Creating the vocabulary for character trigrams...")
            main_vocab = build_vocab(params, tokens=ct_tokens)
    elif params.TOKENIZER == "WordTokenizer":
        main_vocab = word_vocab
    else:
        raise ValueError("No such Tokenizer defined! Check the tokenizer.py for defined tokenizers.")
        
    print("Vocabulary ready, moving on.")
    print("Generating dataloaders - input/output pairs for the CBOW model.")
    if params.TOKENIZER == "WordTokenizer":
        train_data = CustomData(train_file_path, main_vocab, params)
        val_data = CustomData(val_file_path, main_vocab, params)
    else:
        train_data = CustomData(train_file_path, main_vocab, params, word_vocab=word_vocab)
        val_data = CustomData(val_file_path, main_vocab, params, word_vocab=word_vocab)

    print("Training the model.")
    model = Model(main_vocab, params).to(params.DEVICE)
    optimizer = torch.optim.Adam(params=model.parameters())

    if params.LOAD_CHECKPOINT and os.path.exists(params.CHECKPOINT_PATH):
        checkpoint = torch.load(params.CHECKPOINT_PATH, params.DEVICE)
        model.load_state_dict(checkpoint.get("model_state_dict"))
        optimizer.load_state_dict(checkpoint.get("optimizer_state_dict"))
        print("Loaded the previous checkpoint.")
        print("Last train loss: ", checkpoint.get("last_train_loss"))
        print("Last validation loss: ", checkpoint.get("last_validation_loss"))

    trainer = Trainer(model, params, main_vocab, optimizer, train_data, val_data)
    trainer.train()
    trainer.save()
