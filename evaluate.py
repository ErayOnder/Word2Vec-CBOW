# TODO: implement generic evaluation script for your embeddings -
#  it should be extendible to other embedding formats just by updating the load_embeddings function
import os
import json
import argparse
import numpy as np

import torch
from tokenizers import Tokenizer
from scipy.spatial.distance import cosine

from src.params import Parameters
from src.tokenizer import CharacterTrigrams

def load_embeddings(*args, **kwargs):
    tokenizer_name = args[0]
    file_name = tokenizer_name + "_embeddings.txt"
    input_file = os.path.join("embeddings", file_name)
    vocab = {}
    embeddings = []
    with open(input_file, 'r') as f:
        for index, line in enumerate(f):
            parts = line.strip().split()
            word = parts[0]
            vector = [float(v) for v in parts[1:]]
            vocab[word] = index
            embeddings.append(vector)

    embeddings_tensor = torch.tensor(embeddings)
    return embeddings_tensor, vocab

def load_embeddings_from_pt(*args, **kwargs):
    tokenizer_name = args[0]
    embed_file_name = tokenizer_name + "_latest_embeddings.pt"
    embed_file_path = os.path.join("embeddings", embed_file_name)

    embedding_matrix = torch.load(embed_file_path)

    vocab_file_name = tokenizer_name + "_vocab.json"
    vocab_file_path = os.path.join("vocab", vocab_file_name)

    if tokenizer_name == "BytePairEncoding":
        vocab = Tokenizer.from_file(vocab_file_path).get_vocab()
    else:
        with open(vocab_file_path, 'r', encoding='utf-8') as file:
            vocab = json.load(file)
    
    return embedding_matrix, vocab

def tok2id(tokenizer, tok):
    tok_seq = tokenizer.encode(tok).tokens
    return [tokenizer.token_to_id(tok) for tok in tok_seq]

def normalize_embeddings(emb_matrix):
    embedding_matrix = emb_matrix.cpu().detach().numpy()
    norm_factor = np.sqrt(np.sum(embedding_matrix ** 2, axis=1))[:, None]
    return embedding_matrix / norm_factor

def get_n_similar_words(embedding_matrix, vocab, n, word=None, vector=None):
    params = Parameters()
    if word:
        word_idx = vocab.get(word, 0)
        if word_idx < len(params.SPECIALS):
            if word_idx == 0:
                print("The word is out of vocabulary.")
            elif word_idx == 1:
                print("The word is a padding token.")
            else:
                print("The word is a special token.")
            return None
        word_vec = embedding_matrix[word_idx][:,None] 
    else:
        word_vec = vector

    distances = np.matmul(embedding_matrix, word_vec).flatten()
    if word:
        top_n_closest = np.argsort(-distances)[1:n+1]
    else:
        top_n_closest = np.argsort(-distances)[:n]
    top_n_words = {}
    for idx in top_n_closest:
        word = list(vocab.keys())[idx]
        top_n_words[word] = distances[idx]
    return top_n_words

def get_similarity(embedding_matrix, vocab, dist, word1=None, word2=None, word1_vec=None, word2_vec=None):
    params = Parameters()
    if word1 and word2:
        word1_idx = vocab.get(word1, 0)
        word2_idx = vocab.get(word2, 0)
        if word1_idx < len(params.SPECIALS) or word2_idx < len(params.SPECIALS):
            if word1_idx == 0 or word2_idx == 0:
                print("One or both of the words are out of vocabulary.")
            elif word1_idx == 1 or word2_idx == 1:
                print("One or both of the words are padding tokens.")
            else:
                print("One or both words are special tokens.")
            return None
        w1_vec, w2_vec = embedding_matrix[word1_idx], embedding_matrix[word2_idx]
    else: 
        w1_vec, w2_vec = word1_vec, word2_vec
    if dist == "cos":
        return cosine(w1_vec, w2_vec)
    elif dist == "euc":
        return np.linalg.norm(w1_vec - w2_vec)
    else:
        raise ValueError("No such distance defined. Only use 'cos' for cosine or 'euc' for euclidean distance.")

def get_accuracy(embedding_matrix, vocab, tokenizer_name, dist_mes):
    params = Parameters()
    test_syntactic_data_path = "data/SynAnalogyTr.txt"
    test_semantic_data_path = "data/turkish-analogy-semantic.txt"
    if tokenizer_name == "BytePairEncoding":
        tokenizer = Tokenizer.from_file("embeddings/BytePairEncoding_vocab.json")

    total, correct = 0, 0
    with open(test_syntactic_data_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.split()
            if len(line) == 4:
                if tokenizer_name == "WordTokenizer":
                    ta, tb, tc, td = vocab.get(line[0], 0), vocab.get(line[1], 0), vocab.get(line[2], 0), vocab.get(line[3], 0)
                    if ta < len(params.SPECIALS) or tb < len(params.SPECIALS) or tc < len(params.SPECIALS) or td < len(params.SPECIALS):
                        continue
                    a, b, c, d = embedding_matrix[ta], embedding_matrix[tb], embedding_matrix[tc], embedding_matrix[td] 
                elif tokenizer_name == "CharacterTrigrams":
                    tokenizer = CharacterTrigrams()
                    ta, tb, tc, td = tokenizer.tokenize(line[0]), tokenizer.tokenize(line[1]), tokenizer.tokenize(line[2]), tokenizer.tokenize(line[3])
                    a, b, c, d = np.sum([embedding_matrix[vocab.get(id, 0)] for id in ta], axis=0), np.sum([embedding_matrix[vocab.get(id, 0)] for id in tb], axis=0), np.sum([embedding_matrix[vocab.get(id, 0)] for id in tc], axis=0), np.sum([embedding_matrix[vocab.get(id, 0)] for id in td], axis=0)
                    threshold = 0.35 if dist_mes == "cos" else 0.5
                else:
                    ta, tb, tc, td = tok2id(tokenizer, line[0]), tok2id(tokenizer, line[1]), tok2id(tokenizer, line[2]), tok2id(tokenizer, line[3])
                    a, b, c, d = np.sum([embedding_matrix[id] for id in ta], axis=0), np.sum([embedding_matrix[id] for id in tb], axis=0), np.sum([embedding_matrix[id] for id in tc], axis=0), np.sum([embedding_matrix[id] for id in td], axis=0)
                    threshold = 0.4 if dist_mes == "cos" else 0.5
                
                target_vec = b - a + c
                if tokenizer_name == "WordTokenizer":
                    top_5_dict = get_n_similar_words(embedding_matrix, vocab, 5, vector=target_vec)
                    if line[3] in top_5_dict.keys():
                        correct += 1
                else:
                    if target_vec.ndim == 1 and d.ndim == 1:
                        dist = get_similarity(embedding_matrix, vocab, dist_mes, word1_vec=target_vec, word2_vec=d)
                        if dist <= threshold:
                            correct += 1
                total += 1
    syn_accuracy = correct / total
    
    sem_accuracy = {"aile":0, "es-anlamlilar":0, "ülke-başkent":0, "capital-world":0, "para-birimi":0, "sehir-bolge":0, "zit-anlamlilar":0}
    with open(test_semantic_data_path, 'r', encoding="utf-8") as file:
        for line in file:
            line = line.split()
            if line[1] in sem_accuracy.keys() and len(line) == 2:
                if line[1] == "aile":
                    category = line[1]
                    total, correct = 0, 0
                    continue
                else:
                    sem_accuracy[category] = correct / total
                    total, correct = 0, 0
                    category = line[1]
                    continue
            if tokenizer_name == "WordTokenizer":
                ta, tb, tc, td = vocab.get(line[0], 0), vocab.get(line[1], 0), vocab.get(line[2], 0), vocab.get(line[3], 0)
                if ta < len(params.SPECIALS) or tb < len(params.SPECIALS) or tc < len(params.SPECIALS) or td < len(params.SPECIALS):
                    continue
                a, b, c, d = embedding_matrix[ta], embedding_matrix[tb], embedding_matrix[tc], embedding_matrix[td]
            elif tokenizer_name == "CharacterTrigrams":
                tokenizer = CharacterTrigrams()
                ta, tb, tc, td = tokenizer.tokenize(line[0]), tokenizer.tokenize(line[1]), tokenizer.tokenize(line[2]), tokenizer.tokenize(line[3])
                a, b, c, d = np.sum([embedding_matrix[vocab.get(id, 0)] for id in ta], axis=0), np.sum([embedding_matrix[vocab.get(id, 0)] for id in tb], axis=0), np.sum([embedding_matrix[vocab.get(id, 0)] for id in tc], axis=0), np.sum([embedding_matrix[vocab.get(id, 0)] for id in td], axis=0)
                threshold = 0.35 if dist_mes == "cos" else 0.5
            else:
                ta, tb, tc, td = tok2id(tokenizer, line[0]), tok2id(tokenizer, line[1]), tok2id(tokenizer, line[2]), tok2id(tokenizer, line[3])
                a, b, c, d = np.sum([embedding_matrix[id] for id in ta], axis=0), np.sum([embedding_matrix[id] for id in tb], axis=0), np.sum([embedding_matrix[id] for id in tc], axis=0), np.sum([embedding_matrix[id] for id in td], axis=0)
                threshold = 0.4 if dist_mes == "cos" else 0.5
            input_vec = b - a + c
            if tokenizer_name == "WordTokenizer":
                top_5_dict = get_n_similar_words(embedding_matrix, vocab, 5, vector=input_vec)
                if line[3] in top_5_dict.keys():
                    correct += 1
            else:
                if target_vec.ndim == 1 and d.ndim == 1:    
                    dist = get_similarity(embedding_matrix, vocab, dist_mes, word1_vec=target_vec, word2_vec=d)
                    if dist < threshold:
                        correct += 1
            total += 1

    return syn_accuracy, sem_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("tokenizer_name", type=str, help="Name of the tokenizer")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    parser_get_similarity = subparsers.add_parser("get_similarity", help="Get similarity between two words")
    parser_get_similarity.add_argument("word1", type=str, help="The first word")
    parser_get_similarity.add_argument("word2", type=str, help="The second word")
    parser_get_similarity.add_argument("dist_mes", type=str, help="The distance measure: 'cos' or 'euc'")

    parser_get_top_n = subparsers.add_parser("get_top_n", help="Get top n similar words to a given word")
    parser_get_top_n.add_argument("n", type=int, help="Number of top similar words to retrieve")
    parser_get_top_n.add_argument("word", type=str, help="The word to compare")

    parser_get_accuracy = subparsers.add_parser("get_accuracy", help="Get accuracy of the word embeddings")
    parser_get_accuracy.add_argument("dist_mes", type=str, help="The distance measure: 'cos' or 'euc'")
    
    args = parser.parse_args()
    print(args.tokenizer_name)

    emb_matrix, vocab = load_embeddings(args.tokenizer_name)
    emb_matrix = normalize_embeddings(emb_matrix)
    params = Parameters()

    if args.command == "get_similarity":
        print(f"Getting similarity between {args.word1} and {args.word2} with {args.tokenizer_name} tokenizer trained embeddings.")
        if args.tokenizer_name == "CharacterTrigrams":
            tokenizer = CharacterTrigrams()
            w1, w2 = tokenizer.tokenize(args.word1), tokenizer.tokenize(args.word2)
            v1, v2 = np.sum([emb_matrix[vocab.get(id, 0)] for id in w1], axis=0), np.sum([emb_matrix[vocab.get(id, 0)] for id in w2], axis=0)
            similarity = get_similarity(emb_matrix, vocab, args.dist_mes, word1_vec=v1, word2_vec=v2)
        elif args.tokenizer_name == "BytePairEncoding":
            tokenizer = Tokenizer.from_file("embeddings/BytePairEncoding_vocab.json")
            w1, w2 = tok2id(tokenizer, args.word1), tok2id(tokenizer, args.word2)
            v1, v2 = np.sum([emb_matrix[id] for id in w1], axis=0), np.sum([emb_matrix[id] for id in w2], axis=0)
            similarity = get_similarity(emb_matrix, vocab, args.dist_mes, word1_vec=v1, word2_vec=v2)
        else:
            similarity = get_similarity(emb_matrix, vocab, args.dist_mes, word1=args.word1, word2=args.word2)
        print(f"Distance: {similarity}")
    elif args.command == 'get_top_n':
        print(f"Getting top {args.n} similar words to {args.word} with {args.tokenizer_name} tokenizer trained embeddings.")
        top_n_word_dict = get_n_similar_words(emb_matrix, vocab, args.n, word=args.word)
        for key, value in top_n_word_dict.items():
            print(f"{key}: {value}")
    elif args.command == 'get_accuracy':
        print(f"Getting accuracy of word embeddings.")
        syn_acc, sem_acc = get_accuracy(emb_matrix, vocab, args.tokenizer_name, args.dist_mes)
        print(f"Accuracy on Syntatic Dataset: {syn_acc}")
        print(f"Accuracy on Semantic Dataset:")
        for key, val in sem_acc.items():
            print(f"Category: {key} ==> Accuracy: {val}")