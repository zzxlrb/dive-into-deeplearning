import torch
import torch.nn.functional as F
from torch import nn


def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return f'cuda:{i}'
    return torch.device('cpu')


def try_all_gpus():
    cuda_list = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return cuda_list if cuda_list else [torch.device('cpu')]


"""
在这种情况下，两种写法的效果是相同的。torch.device(f'cuda:{i}') 创建了一个表示 CUDA 设备的 torch.device 对象，而 f'cuda:{i}' 创建了一个 CUDA 设备的名称字符串。
在此上下文中，两者都可以作为设备的标识符使用。通常来说，如果你需要明确地创建一个设备对象，使用 torch.device() 是更好的做法。如果只是需要设备名称的字符串表示，可以直接使用字符串字面量。
"""

x = torch.rand(size=(10, 2))
# 默认情况下，张量是在cpu上创建的
print(x.device)

# 数据复制
x = x.to(try_gpu())
# cuda:0和cuda是等价的
print(x)
# 将变量转移至GPU中存储


net = nn.Sequential(nn.Linear(2, 10), nn.ReLU(), nn.Linear(10, 1))

net = net.cuda()
"""
    在创建nn.Sequential类实例对象的时候，不能在其初始化函数中指定模型变量的位置
    但可在之后调用 to('cuda/cuda:0') / cuda() 喊出将模型参数转移至GPU上
"""

# 需要注意的是，在计算过程中，该计算所需用到的所有元素都应在同意设备上，否则引发运行错误
print(net(x))
