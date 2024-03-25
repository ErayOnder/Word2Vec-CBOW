import re
import os
from typing import Union, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from src.params import Parameters

class WordTokenizer:
    def __init__(self, stopwords = None):
        if stopwords is None:
            stopwords = set()
        self.stopwords = set(stopword.lower() for stopword in stopwords)

    def tokenize(self, text):
        text = re.sub(r'http\S+|www.\S+|\S*@\S*\s?', '', text)
        text = text.lower()
        text = re.sub(r'[^a-z\söçşığü]', '', text)
        tokens = text.split()
        if len(self.stopwords) != 0:
            tokens = [token for token in tokens if token not in self.stopwords]  
        return tokens

class CharacterTrigrams:
    def __init__(self, stopwords = None):
        self.tokenizer = WordTokenizer(stopwords)

    def extract_trigrams(self, word):
        word = '<' + word + '>'
        trigrams = [word[i:i+3] for i in range(len(word)-2)]
        return trigrams
    
    def tokenize(self, text):
        words = self.tokenizer.tokenize(text)
        tokens = []
        for word in words:
            tokens.extend(self.extract_trigrams(word))
        return tokens 

class BytePairEncoding:
    def __init__(self, params: Parameters):
        self.params = params
        self.tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        self.tokenizer.pre_tokenizer = Whitespace()
        if os.path.exists(self.params.VOCAB_PATH):
            print("File exists. Loading the BPE tokenizer...")
            self.tokenizer = Tokenizer.from_file(self.params.VOCAB_PATH)
        else:
            self.trainer = BpeTrainer(special_tokens= params.SPECIALS, vocab_size= params.VOCAB_SIZE)
            print("Training BPE tokenizer on this file: ", self.params.TRAIN_DATA_PATH)
            self.tokenizer.train(files=[self.params.TRAIN_DATA_PATH], trainer=self.trainer)
            self.tokenizer.save(self.params.VOCAB_PATH)
            print("BytePairEncoding Vocabulary file written.")
        self.vocab = self.tokenizer.get_vocab()
        
    def tok2id(self, tok):
        tok_seq = self.tokenizer.encode(tok).tokens
        return [self.tokenizer.token_to_id(tok) for tok in tok_seq]
    
    def id2tok(self, id: Union[int, List]):
        if isinstance(id, int):
            return self.tokenizer.id_to_token(id)
        else:
            word = [self.tokenizer.id_to_token(i) for i in id]
            return "".join(word)
    
    def id2seq(self, seq):
        return [self.id2tok(id) for id in seq]
