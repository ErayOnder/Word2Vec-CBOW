import numpy as np
import random
from tqdm import tqdm

import torch
from torch.utils.data import Dataset

from src.tokenizer import WordTokenizer
from src.vocab import Vocab
from src.params import Parameters

class CustomData(Dataset):
    def __init__(self, file_path, main_vocab: Vocab, params: Parameters, word_vocab: Vocab=None, limit = None):
        self.file_path = file_path
        self.main_vocab = main_vocab
        self.params = params
        self.tokenizer = WordTokenizer()
        self.word_vocab = word_vocab
        self.limit = limit
        if self.params.TOKENIZER == "WordTokenizer":
            self.discard_probs = self.discard_dict(self.main_vocab)
            self.ns_array = self.negative_sample_array(self.main_vocab)
        else:
            self.discard_probs = self.discard_dict(self.word_vocab)
            self.ns_array = self.negative_sample_array(self.word_vocab)
        self.load_texts()
         

    def load_texts(self):
        self.contexts, self.targets = [], []
        with open(self.file_path, 'r', encoding="utf-8") as file:
            print("Loading, tokenizing, preprocessing the file: ", self.file_path)
            print("Maximum number of lines to read and tokenize: ", self.limit)
            for i, line in enumerate(tqdm(file)):
                if not line.startswith("=="):
                    tokens = self.tokenizer.tokenize(line.strip())
                    self.get_cbow(tokens)
                if self.limit:
                    if i >= self.limit: break
    
    def get_cbow(self, word_seq):
        for i in range(len(word_seq) - 2*self.params.CBOW_N_WORDS):
            seq = word_seq[i:(i + 2*self.params.CBOW_N_WORDS + 1)]
            output = seq.pop(self.params.CBOW_N_WORDS)
            inputs = seq
            neg_samples = self.neg_sample()
            if self.params.TOKENIZER == "WordTokenizer":
                v = self.main_vocab
                pad = None
            else:
                v = self.word_vocab
                pad = True
            if output in v.get_vocab():
                sub_sample_prob = random.random()
                output_prob = self.discard_probs.get(output, -1)
                if output_prob != -1 or output_prob < sub_sample_prob:
                    for i, input in enumerate(inputs):
                        sub_sample_prob = random.random()
                        input_prob = self.discard_probs.get(input, -1)
                        if input_prob == -1 or input_prob >= sub_sample_prob:
                            inputs[i] = "[PAD]"
                    self.contexts.append(inputs)
                    self.targets.append([output]+neg_samples)
 
    def discard_dict(self, vocab: Vocab):
        freq_list = []
        freq_list.extend([val[1]/vocab.total_freq 
                          for val in list(vocab.id_to_tok_with_freq.values())[len(self.params.SPECIALS):]])
        t = np.percentile(freq_list, self.params.SUBSAMPLE_T)
        discard_dict = {}
        for _,(word, freq) in list(vocab.id_to_tok_with_freq.items())[len(self.params.SPECIALS):]:
            discard_prob = 1 - np.sqrt(t / (freq/vocab.total_freq + t))
            discard_dict[word] = discard_prob
        return discard_dict
    
    def negative_sample_array(self, vocab: Vocab):
        print("Creating negative sampling array...")
        adj_freq_dict = {word: 
                         max(1,int((freq**self.params.NEG_SAMPLE_EXP / vocab.total_freq)*self.params.NS_ARRAY_LEN)) 
                         for _, (word, freq) in list(vocab.id_to_tok_with_freq.items())[len(self.params.SPECIALS):]}
        ns_array = []
        for word, freq in adj_freq_dict.items():
            ns_array.extend([word]*freq)
        random.shuffle(ns_array)
        if self.params.NS_ARRAY_LEN:
            if len(ns_array) > self.params.NS_ARRAY_LEN:
                ns_array = ns_array[:self.params.NS_ARRAY_LEN]
        return ns_array
    
    def neg_sample(self):
        return random.sample(self.ns_array, self.params.NEG_SAMPLES)
    
    def __len__(self):
        return len(self.contexts)
    
    def __getitem__(self, index):
        if self.params.TOKENIZER == "WordTokenizer":                    
            c = self.main_vocab.get_index(self.contexts[index]) 
            t = self.main_vocab.get_index(self.targets[index])
        else:
            c = self.main_vocab.get_index(self.contexts[index], padding=True)
            t = self.main_vocab.get_index(self.targets[index], padding=True)
        return torch.as_tensor(np.array(c)), torch.as_tensor(np.array(t))
