import torch
from d2l import torch as d2l
import torch.nn as nn

def corr2d_multi_in(X,K):
    return sum(d2l.corr2d(x,k) for x,k in zip(X,K))

def corr2d_multi_out(X,K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

