import random
import re
from collections import Counter
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from d2l import torch as d2l
import matplotlib.pyplot as plt
import numpy as np
import jieba
import os
from gensim.models import KeyedVectors
from datasets import Dataset
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

device = "cuda"
batch_size = 32


def loda_data(neg, pos):
    comments = []
    for name in os.listdir(neg):
        with open(neg + "/" + name, 'r', encoding="utf-8") as f:
            text = f.readline()
            text = re.sub("[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", text)
            text = [0, text]
            comments.append(text)
    neg_examples = len(comments)
    for name in os.listdir(pos):
        with open(pos + "/" + name, 'r', encoding="utf-8") as f:
            text = f.readline()
            text = re.sub("[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", text)
            text = (1, text)
            comments.append(text)
    random.shuffle(comments)
    all_examples = len(comments)
    pos_examples = all_examples - neg_examples
    print(f'共计读入{all_examples}个样本，{pos_examples}个正例样本，{neg_examples}个负例样本')
    return comments


def split_data(split, data):
    all_examples = len(data)
    split_line = int(all_examples * split)
    train_comments, train_labels = [t[1] for t in data[: split_line]], [[t[0]] for t in data[: split_line]]
    test_comments, test_labels = [t[1] for t in data[split_line:]], [[t[0]] for t in data[split_line:]]
    return train_comments, train_labels, test_comments, test_labels


def cut(comments):
    cut_list = []
    for comment in comments:
        cut_sentence = []
        cut = jieba.cut(comment)
        for i in cut:
            cut_sentence.append(i)
        cut_list.append(cut_sentence)
    return cut_list


neg_path = "../data/datanew/neg"
pos_path = "../data/datanew/pos"

total_data = loda_data(neg_path, pos_path)
all_comments = [t[1] for t in total_data]
all_lables = [[t[0]] for t in total_data]

train_comments, train_labels, test_comments, test_labels = split_data(0.8, total_data)
train_comments = cut(train_comments)
test_comments = cut(test_comments)
# 接下来需要进行token化
cn_model = KeyedVectors.load_word2vec_format('./sgns.zhihu.bigram')


# words = Counter()
#
# for i, comment in enumerate(all_comments):
#     words.update(comment)
#
# words = {k: v for k, v in words.items() if v > 1}
#
# words = sorted(words, key=words.get, reverse=True)
#
# words = ['<PAD>'] + words
#
# word2idx = {v: k for k, v in enumerate(words)}
# idx2word = {k: v for k, v in enumerate(words)}


def tokenize(comments, length):
    tokens = []
    for comment in comments:
        token = []
        for i, word in enumerate(comment):
            try:
                token.append(cn_model.key_to_index[word])
            except KeyError:
                token.append(0)
        sentence_len = len(comment)
        if sentence_len > length:
            tokens.append(token[:length])
        else:
            tokens.append(token + [0] * (length - sentence_len))
    return torch.tensor(tokens)


train_comments = tokenize(train_comments, 50)
test_comments = tokenize(test_comments, 50)

num_words = len(cn_model.key_to_index)
embedding_dim = 300

# 初始化embedding_matrix
embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i, :] = cn_model[cn_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')

train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)
train_data = TensorDataset(train_comments, train_labels)
test_data = TensorDataset(test_comments, test_labels)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


class LstmModel(nn.Module):
    def __init__(self, embedding_dim, embedding_matrix, hidden_size):
        super(LstmModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix), freeze=False)

        self.rnn = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        out, _ = self.rnn(embedded)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out


model = LstmModel(embedding_dim, embedding_matrix, 32)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.5)


def train(net, train_loader, test_loader, criterion, optimizer, epochs, device="cuda"):
    net.to(device)
    trainloss_list = []
    testloss_list = []
    train_accuracy = []
    test_accuracy = []
    for epoch in range(epochs):
        total_loss = 0.0
        sum_correct = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs.float(), y.float()).to(device)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            sum_correct += torch.sum(outputs == y).item()
        trainloss_list.append(total_loss)
        print(f'total_loss :{total_loss}')
        train_accuracy.append(sum_correct / len(train_loader.dataset))
        total_loss = 0.0
        sum_correct = 0
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            test_loss = criterion(outputs.float(), y.float()).to(device)
            total_loss += test_loss.item()
            outputs = (outputs>0.5).int().to(device)
            sum_correct += torch.sum((outputs == y).int()).item()
        testloss_list.append(total_loss)
        test_accuracy.append(sum_correct / len(test_loader.dataset))
    print(trainloss_list, test_accuracy)


train(model, train_loader, test_loader, criterion, optimizer, 10)
