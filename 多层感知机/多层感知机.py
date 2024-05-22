import torch
import numpy as np
import matplotlib.pyplot as plt
import tools
from d2l import torch as d2l
x=torch.arange(-10.0,10.0,0.01,requires_grad=True)
y=torch.relu(x)
x.grad.data.zero_()

plt.plot(x.detach(),y.detach())
plt.show()