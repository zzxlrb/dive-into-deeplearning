import torch
from torch import nn

net = nn.Sequential(nn.Linear(74, 8), nn.ReLU(), nn.Linear(8, 1))

# 参数访问
# print(net.state_dict())
"""
    state_dict()：
        该方法返回一个字典，其中包含了指定层的所有参数张量的键值对。
        键是参数的名称，而值是参数张量本身。
    parameters()：
        该方法获取模型中所有可学习参数的生成器，通常用于定义优化器和计算损失函数时传递模型参数
"""
# print(*[(name, param.shape) for name, param in net.named_parameters()])
"""
    named_parameters()：
        该方法的功能是获取神经网络模型中所有参数以及它们的名称。
        返回一个可迭代对象，其中每个元素都是一个元组，包含参数的并称和对应的参数张量
"""


def block1():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                         nn.Linear(8, 4), nn.ReLU())


def block2():
    net = nn.Sequential()
    for i in range(4):
        # 在这里嵌套
        net.add_module(f'block {i}', block1())
        # 这里是向模型中添加块。函数的第一个参数是块名称，第二个参数是块类型
        # nn.Module中实际的存储方式亦是分块来处理的
    return net


"""
    格式化字符串：
        以 f 或 F 开头的字符串被称为 f-string。在 f-string 中，
        使用大括号 {} 将要插入的变量或表达式括起来。在 {} 中可以放置变量、表达式、函数调用等，
        它们会在运行时被求值，并将结果插入到字符串中。
"""

# 参数绑定
shared = nn.Linear(8, 8)
net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(),
                    shared, nn.ReLU(),
                    shared, nn.ReLU(),
                    nn.Linear(8, 1))
# 在这个例子中，第三层和第五层的神经网络层参数是绑定的，它们不仅值相等，而且有相同的张良表示
# 需要注意的是。由于神经网络层参数是绑定的，他们的对应的梯度也会共享，这就意味着，在反向传播过程中，多个参数的梯度会相互累加

