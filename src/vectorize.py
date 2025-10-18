import torch
from typing import List, Dict

def build_vocab(list_of_ingredient_lists: List[List[str]]):
    vocab = {}
    for lst in list_of_ingredient_lists:
        for w in lst:
            if w not in vocab:
                vocab[w] = len(vocab)
    return vocab

def vectorize(items: List[List[str]], vocab: Dict[str, int], weights: Dict[str, float] = None):
    D = len(vocab)
    mat = torch.zeros(len(items), D, dtype=torch.float32)
    for i, lst in enumerate(items):
        for w in lst:
            if w in vocab:
                j = vocab[w]
                mat[i, j] += (1.0 if weights is None else float(weights.get(w, 1.0)))
    return mat
