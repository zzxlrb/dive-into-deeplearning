import torch
from d2l import torch as d2l

X, W_xh = torch.normal(0, 1, (3, 1)), torch.normal(0, 1, (1, 4))
H, W_hh = torch.normal(0, 1, (3, 4)), torch.normal(0, 1, (4, 4))
torch.matmul(X, W_xh) + torch.matmul(H, W_hh)
print(X, H)

print(torch.cat((X, H), 1))
# 得出的结论是可以进行变量之间的拼接（按行）以及参数矩阵的拼接（按列）
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))