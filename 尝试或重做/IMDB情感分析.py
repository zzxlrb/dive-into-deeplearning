from d2l import torch as d2l
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn


def tokenize(filename):
    vocab = []
    with open(filename, 'r') as f:
        text = f.read()
        print(text)
        vocab.append(text)


filename = "../data/aclImdb/imdb.vocab"

vocab = d2l.Vocab(tokenize(filename), reserved_tokens=["PAD", "STA", "UNK"])
