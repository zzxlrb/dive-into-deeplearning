import os
import random
import re
import jieba
import warnings
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from tqdm import tqdm
from sklearn.metrics import recall_score, roc_curve
import matplotlib.pyplot as plt

device = "cuda"

pos_dir = "../data/datanew/pos"
neg_dir = "../data/datanew/neg"

s_length = 100
batch_size = 32

train_loss, test_loss = [], []
train_acc, test_acc = [], []
score = []
lable = []


def load_all_comment(pos_dir, neg_dir):  # 读取文件内容
    all_comment = []
    for filename in os.listdir(pos_dir):
        with open(pos_dir + "/" + filename, 'r', encoding='UTF-8') as f:
            comment = f.read().strip()
            all_comment.append(comment)
            f.close()
    for filename in os.listdir(neg_dir):
        with open(neg_dir + "/" + filename, 'r', encoding='UTF-8') as f:
            comment = f.read().strip()
            all_comment.append(comment)
            f.close()
    return all_comment


all_comment = load_all_comment(pos_dir, neg_dir)

pre_model = KeyedVectors.load_word2vec_format('./sgns.zhihu.bigram')  # 加载预训练的embedding矩阵


def tokenize(comments):  # 该函数的作用是在去除所有非中文字符后将字符token化
    tokens = []
    for comment in comments:
        comment = re.sub("[\s+\.\!\/_,-|$%^*(+\"\')]+|[+——！，； 。？ 、~@#￥%……&*（）]+", "", comment)
        cut_list = jieba.cut(comment)
        token = []
        for word in cut_list:
            try:
                token.append(pre_model.key_to_index[word])
            except KeyError:
                token.append(0)
        tokens.append(token)
    return tokens


all_tokens = tokenize(all_comment)

num_words = len(pre_model.key_to_index)
embedding_dim = 300

embedding_matrix = np.zeros((num_words, embedding_dim))
for i in range(num_words):
    embedding_matrix[i, :] = pre_model[pre_model.index_to_key[i]]
embedding_matrix = embedding_matrix.astype('float32')


def cut_padding(tokens, length):  # 进行句子的裁切
    process_tokens = []
    for token in tokens:
        l = len(token)
        if (l < length):
            token.extend([0] * (length - l))
        else:
            token = token[:length]
        process_tokens.append(token)
    return np.array(process_tokens)


all_tokens = cut_padding(all_tokens, s_length)

all_labels = np.array([1 for i in range(2000)] + [0 for i in range(2000)])  # 生成对应的lable集

train_num = 3600
total_num = 2000
sum_num = 4000

idx = [id for id in range(sum_num)]
idx = torch.randperm(len(idx))  # 生成标签序列并打乱用于抽取训练集与测试集

train_labels = np.array(all_labels[idx[:train_num]])
train_tokens = np.array(all_tokens[idx[:train_num]])
test_tokens = np.array(all_tokens[idx[train_num:]])
test_labels = np.array(all_labels[idx[train_num:]])

train_tokens = torch.from_numpy(train_tokens).int()
test_tokens = torch.from_numpy(test_tokens).int()
train_labels = torch.from_numpy(train_labels).long()
test_labels = torch.from_numpy(test_labels).long()

train_data = TensorDataset(train_tokens, train_labels)
test_data = TensorDataset(test_tokens, test_labels)
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


# 加载数据集


class RNN(nn.Module):
    def __init__(self, embedding_matrix, embedding_size, hidden_size, num_layers, drop_out):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix),
                                                      freeze=False)  # 加载预训练好的embedding矩阵
        self.gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, batch_first=True,
                          bidirectional=True)  # 三层GRU层
        self.lstm = nn.LSTM(2 * hidden_size, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=True)  # 三层双向LSTM层
        self.dropout = nn.Dropout(drop_out)  # 添加概率为0.5的Dropout层
        self.linear = nn.Linear(2 * hidden_size, 1)  # 全连接先行曾
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):  # 进行前向传播
        out = self.embedding(X)
        out, _ = self.gru(out)
        out = self.dropout(out)
        out, _ = self.lstm(out)
        out = self.dropout(out)
        out = self.linear(out[:, -1, :])
        out = self.sigmoid(out)
        return out


net = RNN(embedding_matrix, 300, hidden_size=256, num_layers=2, drop_out=0.6)

def correct_num(predictions, labels):  # 用于计算在模型预测过程中预测样例的正确个数
    predictions = (predictions > 0.5).type(torch.uint8)
    correct = (predictions == labels).type(torch.uint8)
    num = correct.sum()
    return num.detach().cpu().numpy().item()


def train(model, train_dataloader, test_dataloader, lr, epochs, device="cuda"):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0.0
        total_correct = 0
        for X, y in train_dataloader:
            net.train()
            X, y = X.to(device), y.to(device)
            prediction = net(X)
            loss = criterion(prediction.squeeze(), y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += correct_num(prediction.squeeze(), y)
        train_loss.append(total_loss)
        train_acc.append(total_correct * 1.0 / 3600)
        print(f'Epoch: {epoch + 1} TrainLoss: {total_loss} ')
        total_loss = 0.0
        total_correct = 0
        for X, y in test_dataloader:
            net.eval()
            X, y = X.to(device), y.to(device)
            prediction = net(X)
            loss = criterion(prediction.squeeze(), y.float())
            total_loss += loss.item()
            total_correct += correct_num(prediction.squeeze(), y)
        test_loss.append(total_loss)
        test_acc.append(total_correct * 1.0 / 400)
        print(f'Epoch: {epoch + 1} TestLoss: {total_loss} ')
        if (total_correct * 1.0 / 400 >= 0.91):
            torch.save(net,'rnn_model')
            return epoch+1
    return epochs

access_data = DataLoader(test_data, 1)


def access(model, access_data, epochs):  # 用于加载模型的的性能指标 如ROC，ACC等
    model.eval()
    total_num = 0
    for X, y in access_data:
        X, y = X.to(device), y.to(device)
        prediction = net(X)
        total_num += correct_num(prediction.squeeze(), y)
        lable.append(y.detach().cpu().item())
        score.append(prediction.squeeze().detach().cpu().numpy().item())
    print(f'acc:{total_num * 1.0 / len(access_data)}')
    recall = recall_score(lable, [1 if x >= 0.5 else 0 for x in score])
    print(f'Recall: {recall}')
    fpr, tpr, thresholds = roc_curve(lable, score)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()
    plt.cla()
    x = [i for i in range(epochs)]
    plt.plot(x, train_acc, 'r', label='Train')
    plt.plot(x, test_acc, 'b', label='Test')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend()
    plt.show()


epochs = 15

epochs = train(net, train_dataloader, test_dataloader, lr=0.00031065, epochs=epochs)
access(net, access_data, epochs=epochs)
