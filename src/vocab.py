import numpy as np
import json
import os

from collections import Counter, OrderedDict
from typing import List, Union

from src.tokenizer import BytePairEncoding, CharacterTrigrams
from src.params import Parameters

def build_vocab(params: Parameters, tokens=None, tokenizer=None, word_vocab_path=None):
    if tokens and not tokenizer:
        counter = Counter(tokens)
        ordered_dict = OrderedDict(sorted(counter.items(), key=lambda x: (-x[1], x[0])))
        specials = [(special, np.nan) for special in params.SPECIALS]
        tokens_with_freq = [special for special in specials]
        for token, freq in ordered_dict.items():
            if freq < params.MIN_FREQ or len(tokens_with_freq) >= params.VOCAB_SIZE:
                break
            tokens_with_freq.append((token, freq))
        if word_vocab_path:
            return Vocab(params, list=tokens_with_freq, word_vocab_path=word_vocab_path)
        return Vocab(params, list=tokens_with_freq)
    elif tokenizer and not tokens:
        return Vocab(params, tokenizer=tokenizer)
    else:
        raise ValueError("Wrong set of parameter are entered for build_vocab function, either pass on the tokens or the tokenizer!")

class Vocab:
    def __init__(self, params: Parameters, list=None, tokenizer=None, word_vocab_path=None):
        self.params = params
        if list and not tokenizer:
            self.bpe = False
            self.tokenizer = None
            self.id_to_tok_with_freq = {i: (tup[0], tup[1]) for i, tup in enumerate(list)}
            self.total_freq = np.nansum([freq for _,(_,freq) in self.id_to_tok_with_freq.items()], dtype=int)
            if word_vocab_path:
                if os.path.exists(word_vocab_path):
                    self.vocab = self.load_vocab(word_vocab_path)
                else:
                    self.vocab = {tup[0]: i for i, tup in enumerate(list)}
                    self.save_vocab(word_vocab_path)
            else:
                if os.path.exists(self.params.VOCAB_PATH):
                    self.vocab = self.load_vocab(self.params.VOCAB_PATH)
                else:
                    self.vocab = {tup[0]: i for i, tup in enumerate(list)}
                    self.save_vocab(self.params.VOCAB_PATH)
        elif tokenizer and not list:
            self.tokenizer = tokenizer
            self.id_to_tok_with_freq = None
            self.total_freq = None
            if self.params.TOKENIZER == "BytePairEncoding":
                self.bpe = True
                self.vocab = self.tokenizer.vocab
            else:
                self.bpe = False
                self.vocab = self.load_vocab(self.params.VOCAB_PATH)
        else:
            raise ValueError("Wrong set of parameter are entered for Vocab class")
        
    def get_index(self, word: Union[str, List], padding=None):
        if isinstance(word, str):
            if self.params.TOKENIZER == "WordTokenizer":
                return self.vocab.get(word, 0)
            elif self.bpe:
                if padding:
                    return self.pad_or_trunc(self.tokenizer.tok2id(word))
                return self.tokenizer.tok2id(word) 
            else:
                if word in self.params.SPECIALS:
                    if padding:
                        return self.pad_or_trunc([self.vocab.get(word)])
                    return [self.vocab.get(word)]
                else:
                    tokenizer = CharacterTrigrams()
                    tok_word = tokenizer.tokenize(word)
                    if padding:
                        return self.pad_or_trunc([self.vocab.get(tok, 0) for tok in tok_word])
                    return [self.vocab.get(tok, 0) for tok in tok_word]
        elif isinstance(word, List):
            if self.params.TOKENIZER == "WordTokenizer":
                return [self.vocab.get(w, 0) for w in word]
            else:
                return [self.get_index(w, padding=padding) for w in word]
        else:
            raise ValueError("To get the corresponding index of the token/tokens in the vocabulary, the input should either be a string or a list of strings.")
    
    def get_word(self, id: Union[int, List]):
        if isinstance(id, int):
            if self.bpe:
                return self.tokenizer.id2tok(id)
            else:
                return self.id_to_tok_with_freq.get(id, ("[UNK]", 0))[0]
        elif isinstance(id, List):
            if self.bpe:
                if all(isinstance(item, List) for item in id):
                    return self.tokenizer.id2seq(id)
                else:
                    return self.tokenizer.id2tok(id)
            else:
                result = []
                if self.params.TOKENIZER == "WordTokenizer":
                    for i in id:
                        result.append(self.id_to_tok_with_freq.get(i, ("[UNK]", 0))[0])
                    return result
                else:
                    if all(isinstance(item, List) for item in id):
                        for i in id:
                            word = ""
                            for gram in i:
                                trigram = self.id_to_tok_with_freq.get(gram, ("[UNK]", 0))[0]
                                word += trigram[-2]
                            result.append(word)
                        return result
                    else:
                        word = ""
                        for i in id:
                            trigram = self.id_to_tok_with_freq.get(i, ("[UNK]", 0))[0]
                            word += trigram[-2]
                        return word
        else:
            raise ValueError("To get the corresponding token given its index in the vocabulary, the input should either be a integer or a list of integers.")

    def pad_or_trunc(self, seq):
        if len(seq) >= self.params.MAX_SEQ_LENGHT:
            return seq[:self.params.MAX_SEQ_LENGHT]
        else:
            num_padding = self.params.MAX_SEQ_LENGHT - len(seq)
            padding = [1] * num_padding
            return seq + padding
               
    def save_vocab(self, file_path):
        with open(file_path, 'w') as file:
            json.dump(self.vocab, file, indent=4)

    def load_vocab(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            vocab_dict = json.load(file)
        return vocab_dict
        
    def get_lenght(self):
        return len(self.vocab)
    
    def get_vocab(self):
        return self.vocab
