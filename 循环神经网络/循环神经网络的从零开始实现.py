import math
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import tools

batch_size, num_steps = 32, 35
train_iter, vocab = tools.load_data_time_machine(batch_size, num_steps)
X = torch.arange(10).reshape((2, 5))


def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        # 初始化隐藏层状态
        return torch.randn(size=shape, device=device) * 0.01

    # 隐藏层参数
    W_xh = normal((num_inputs, num_hiddens))
    W_hh = normal((num_hiddens, num_hiddens))
    b_h = torch.zeros(num_hiddens, device=device)
    # 输出层参数
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # 附加梯度
    params = [W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        # 将每个参数设置为需要梯度
        param.requires_grad_(True)
    return params


def init_rnn_state(batch_size, num_hiddens, device):
    # 这里设置位一个二元的tuple主要是为了方便第九章的LSTM填充矩阵C
    return (torch.zeros((batch_size, num_hiddens), device=device),)


def rnn(inputs, state, params):
    # inputs的形状：(时间步数量，批量大小，词表大小)
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []
    # X的形状：(批量大小，词表大小)
    for X in inputs:
        # 这里for X in inputs 遍历input的第一个维度
        # 这里使用tanh作为激活函数有两个原因：1.RNN诞生初期没有ReLU 2.tanh可以将实属域映射到 [-1,1]
        # 等号左边的H是H_t
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        outputs.append(Y)
        # 这里的cat听起来的意思是按高度垒放？不是很清楚
    return torch.cat(outputs, dim=0), (H,)


class RNNModelScratch:
    """从零开始实现的循环神经网络模型"""

    def __init__(self, vocab_size, num_hiddens, device,
                 get_params, init_state, forward_fn):
        self.vocab_size, self.num_hiddens = vocab_size, num_hiddens
        self.params = get_params(vocab_size, num_hiddens, device)
        self.init_state, self.forward_fn = init_state, forward_fn

    def __call__(self, X, state):
        X = F.one_hot(X.T, self.vocab_size).type(torch.float32)
        return self.forward_fn(X, state, self.params)

    def begin_state(self, batch_size, device):
        return self.init_state(batch_size, self.num_hiddens, device)


def predict_ch8(prefix, num_preds, net, vocab, device):
    """在prefix后面生成新字符"""
    state = net.begin_state(batch_size=1, device=device)
    # 将输出结果的第一个元素设置为输入语句中的第一个元素。
    # 注意：输出结果中的元素并非实际的字符/单词，而是对应词典哈希出的一个整数
    outputs = [vocab[prefix[0]]]
    # 每次模型的输入应该是上一轮输出结果，这里的-1表示从后往前取
    get_input = lambda: torch.tensor([outputs[-1]], device=device).reshape((1, 1))
    for y in prefix[1:]:  # 预热期
        """
            在这里因为是对模型进行预热（即对隐藏状态进行“初始化”），并且拥有标准的答案
            所以不关心该模型返回的预测结果，而是着重更新state，并将准确的字符添加至预测结果中
        """
        _, state = net(get_input(), state)
        outputs.append(vocab[y])
    for _ in range(num_preds):  # 预测num_preds步
        # 在更新隐藏状态的同时将预测结果添加至output list中
        y, state = net(get_input(), state)
        # 因为这里对字符进行one-hot编码，所以需要去字符中的最大值作为预测结果
        outputs.append(int(y.argmax(dim=1).reshape(1)))
    return ''.join([vocab.idx_to_token[i] for i in outputs])


num_hiddens = 512
net = RNNModelScratch(len(vocab), num_hiddens, d2l.try_gpu(), get_params,
                      init_rnn_state, rnn)
state = net.begin_state(X.shape[0], d2l.try_gpu())
Y, new_state = net(X.to(d2l.try_gpu()), state)


def grad_clipping(net, theta):
    """
        因为RNN中的隐藏状态是进行累加得到的，为避免梯度爆炸，需要进行梯度剪裁
    """
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def train_epoch_ch8(net, train_iter, loss, updater, device, use_random_iter):
    """训练网络一个迭代周期（定义见第8章）"""
    state, timer = None, d2l.Timer()
    metric = d2l.Accumulator(2)  # 训练损失之和,词元数量
    for X, Y in train_iter:
        if state is None or use_random_iter:
            # 在第一次迭代或使用随机抽样时初始化state，则隐藏层状态需要每次进行预测前置零
            state = net.begin_state(batch_size=X.shape[0], device=device)
        else:
            if isinstance(net, nn.Module) and not isinstance(state, tuple):
                # state对于nn.GRU是个张量
                state.detach_()
            else:
                # state对于nn.LSTM或对于我们从零开始实现的模型是个张量
                for s in state:
                    s.detach_()
        y = Y.T.reshape(-1)
        X, y = X.to(device), y.to(device)
        y_hat, state = net(X, state)
        print(f'y:{y}')
        l = loss(y_hat, y.long()).mean()
        """
            这里的loss函数位CrossEntropyLoss，为什么可以使用这个损失函数呢？
                1. 从上面编写的代码来看，进行RNN预测实际上是在进行多分类问题
                2. 在这里我们将标准的文本序列转置后降低维度以同“int(y.argmax(dim=1).reshape(1))”相匹配
        """
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.backward()
            grad_clipping(net, 1)
            updater.step()
        else:
            l.backward()
            grad_clipping(net, 1)
            # 因为已经调用了mean函数
            updater(batch_size=1)
        metric.add(l * y.numel(), y.numel())
        # 这里的困惑度是交叉熵损失函数的平均值取对数得到的
    return math.exp(metric[0] / metric[1]), metric[1] / timer.stop()


def train_ch8(net, train_iter, vocab, lr, num_epochs, device,
              use_random_iter=False):
    loss = nn.CrossEntropyLoss()
    animator = d2l.Animator(xlabel='epoch', ylabel='perplexity',
                            legend=['train'], xlim=[10, num_epochs])
    # 初始化
    if isinstance(net, nn.Module):
        updater = torch.optim.SGD(net.parameters(), lr)
    else:
        updater = lambda batch_size: d2l.sgd(net.params, lr, batch_size)
    predict = lambda prefix: predict_ch8(prefix, 50, net, vocab, device)
    # 训练和预测
    for epoch in range(num_epochs):
        ppl, speed = train_epoch_ch8(
            net, train_iter, loss, updater, device, use_random_iter)
        if (epoch + 1) % 10 == 0:
            print(predict('time traveller'))
            animator.add(epoch + 1, [ppl])
    print(f'困惑度 {ppl:.1f}, {speed:.1f} 词元/秒 {str(device)}')
    print(predict('time traveller'))
    print(predict('traveller'))


num_epochs, lr = 500, 1

train_ch8(net, train_iter, vocab, lr, num_epochs, d2l.try_gpu())

plt.show()
