import torch
import random

from tinycss2 import tokenizer
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
import os
import re
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, roc_curve
import math

loss_list, train_acc_list, test_acc_list = [], [], []
recall_list = []


def loda_data(neg, pos): #加载数据集
    comments = []
    for name in os.listdir(neg):
        with open(neg + "/" + name, 'r', encoding="utf-8") as f:
            text = f.readline()
            text = re.sub('[?!.,]', ' ', text).strip()
            text = [0, text]
            comments.append(text)
    neg_examples = len(comments)
    for name in os.listdir(pos):
        with open(pos + "/" + name, 'r', encoding="utf-8") as f:
            text = f.readline()
            text = re.sub('[?!.,]', ' ', text).strip()
            text = (1, text)
            comments.append(text)
    random.shuffle(comments)
    all_examples = len(comments)
    pos_examples = all_examples - neg_examples
    print(f'共计读入{all_examples}个样本，{pos_examples}个正例样本，{neg_examples}个负例样本')
    return comments


def split_data(split, data):  # 划分测试集与训练集
    all_examples = len(data)
    split_line = int(all_examples * split)
    train_comments, train_labels = [t[1] for t in data[: split_line]], [t[0] for t in data[: split_line]]
    test_comments, test_labels = [t[1] for t in data[split_line:]], [t[0] for t in data[split_line:]]
    return train_comments, train_labels, test_comments, test_labels


neg_path = "../data/datanew/neg"
pos_path = "../data/datanew/pos"

total_data = loda_data(neg_path, pos_path)

train_comments, train_labels, test_comments, test_labels = split_data(0.9, total_data)


class BERT(nn.Module):  # 定义Bert模型
    def __init__(self, output_dim, input_dim=768, pretrainedd_name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrainedd_name)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        res = self.bert(**X)
        drop_output = self.dropout(res[1])
        logits = self.linear(drop_output)
        return logits


batch_size = 4


def count_num(labels, results):
    cnt = 0
    for i in range(len(results)):
        if labels[i] == results[i]:
            cnt += 1
    return cnt


def evaluate(net, comments, labels, batch_size, tokenizer):
    num_correct, pointer = 0, 0
    net.eval() # 将net设置为评价模式 避免Dropout层的影响
    while pointer < len(comments):
        comment = comments[pointer:min(pointer + batch_size, len(comments))]
        tokens = tokenizer(comment, padding="max_length", truncation=True, return_tensors="pt").to(device)
        result = net(tokens)
        label = labels[pointer:min(pointer + batch_size, len(comments))]
        num_correct += count_num(label, result.argmax(1))
        pointer += batch_size
    return num_correct * 1.0 / len(comments)


def BERT_CLASSIFIER(net, batch_size, tokenizer, train_comments, train_labels, test_comments, test_labels, device,
                    epochs, optimizer, loss): # 用于模型预测
    train_acc = evaluate(net, train_comments, train_labels, batch_size=batch_size, tokenizer=tokenizer)
    test_acc = evaluate(net, test_comments, test_labels, batch_size=batch_size, tokenizer=tokenizer)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f'Initial Train Acc: {train_acc}, Initial Test Acc: {test_acc}')
    for epoch in tqdm(range(epochs)):
        pointer, total_loss = 0, 0
        while pointer < len(train_comments):
            comments = train_comments[pointer:min(pointer + batch_size, len(train_comments))]
            tokens = tokenizer(comments, padding="max_length", truncation=True, return_tensors="pt").to(device)
            results = net(tokens)
            y = torch.tensor(train_labels[pointer: min(pointer + batch_size, len(train_labels))]).reshape(-1).to(
                device=device)
            optimizer.zero_grad()
            loss_value = loss(results, y)
            loss_value.backward()
            optimizer.step()
            total_loss += loss_value.detach().cpu().numpy()
            pointer += batch_size
        train_acc = evaluate(net, train_comments, train_labels, batch_size=batch_size, tokenizer=tokenizer)
        test_acc = evaluate(net, test_comments, test_labels, batch_size=batch_size, tokenizer=tokenizer)
        loss_list.append(total_loss)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(f'Epoch: {epoch + 1}, Loss: {total_loss}, Train Acc: {train_acc}, Test Acc: {test_acc}')
    torch.save(net, 'trained_model.pt')


device = torch.device("cuda")
net = BERT(output_dim=2).to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')  # 加载Bert与训练好的tokenizer

loss = nn.CrossEntropyLoss()

learning_rate = 1e-6

params_1x = [param for name, param in net.named_parameters()
             if name not in ["linear.weight", "linear.bias"]]
optimizer = torch.optim.Adam([{'params': params_1x},
                                   {'params': net.linear.parameters(),
                                    'lr': learning_rate * 100,
                                    'weight_decay':1e-6}],
                                lr=learning_rate)

# 进行微调，对预训练好的Bert模型，学习率为1e-6；而对于线性层，学习率为Bert层学习率的100倍，decay为1e-6

print(optimizer)

# optimizer = torch.optim.Adam(net.parameters(), lr=1e-5, weight_decay=0.01)

# 在这里进行正则化及学习率调整

BERT_CLASSIFIER(net, batch_size, tokenizer, train_comments, train_labels, test_comments, test_labels, device, 10,
                optimizer, loss=loss)

all_comments = [t[1] for t in total_data]
all_lables = [t[0] for t in total_data]


def access(net, tokenizer): # 同样用于加载评测模型的性能指标
    net.eval()
    pre_lable = []
    pre_pro = []
    for text in all_comments:
        token = tokenizer(text, padding="max_length", truncation=True, return_tensors="pt").to(device)
        y_hat = net(token)
        y_hat = torch.exp(y_hat)
        sum = torch.sum(y_hat).item()
        lable = y_hat.argmax(dim=1)[0]
        lable = lable.cpu().numpy()
        value = y_hat[0][lable].item() / sum
        pre_lable.append(lable)
        pre_pro.append(value)
    recall = recall_score(all_lables, pre_lable)
    fpr, tpr, thresholds = roc_curve(all_lables, pre_pro)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.show()
    print(f'Recall: {recall}')
    x = [i for i in range(len(loss_list))]
    plt.plot(x, loss_list, label='Loss', color='red', linestyle='--')
    x = [i for i in range(len(loss_list) + 1)]
    plt.plot(x, train_acc_list, label='Train Accuracy', color='green', linestyle=':')
    plt.plot(x, test_acc_list, label='Test Accuracy', color='blue', linestyle='-')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


access(net, tokenizer)
