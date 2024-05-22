import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

w_history = []
b_history = []
cost_history = []
alpha = 0.03
w = 1
b = 1
m = 10
num = 0


def function(x):
    global w, b
    return w * x + b


def cost_function(x, y):
    return np.sum(np.multiply(function(x) - y, function(x) - y)) / (2 * m)


def p1(x, y):
    return np.sum(np.multiply(function(x) - y, x)) / m


def p2(x, y):
    return np.sum(function(x) - y) / m



def gradient_descent(x, y):
    global w, b, alpha, num
    w_history.append(w)
    b_history.append(b)
    cost_history.append(cost_function(x, y))
    while num < 10000:
        if num > 1 and abs(cost_function(x, y) - cost_history[num - 1]) < 1e-10:
            break
        num += 1
        w = w - alpha * p1(x, y)
        b = b - alpha * p2(x, y)
        w_history.append(w)
        b_history.append(b)
        cost_history.append(cost_function(x, y))


def print_graph(x,y):
    fig,axs =plt.subplots(1,2)
    xpoints = np.array(x)
    ypoints = np.array(y)
    X = np.linspace(0, 20)
    axs[0].plot(X, function(X), "r--")
    axs[0].scatter(xpoints, ypoints)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    step = 0.01
    xline = np.arange(0, 6, step)
    yline = np.arange(0, 15, step)
    x_grid, y_grid = np.meshgrid(xline, yline)
    loss_grid = np.zeros_like(y_grid)
    for j in range(len(xline)):
        for i in range(len(yline)):
            loss_grid[i, j] = np.sum(np.multiply(xline[j]*x+yline[i] - y, xline[j]*x+yline[i] - y)) / (2 * m)
    axs[1].contourf(x_grid, y_grid, loss_grid, levels=50, cmap='viridis')
    axs[1].set_xlabel("w")
    axs[1].set_ylabel("b")
    plt.scatter(w_history, b_history)
    plt.show()

def main():
    x = np.mat([8, 3, 9, 7, 16, 5, 3, 10, 4, 6])
    y = np.mat([30, 21, 35, 27, 42, 24, 10, 38, 22, 25])
    gradient_descent(x, y)
    print_graph(x,y)


if __name__ == '__main__':
    main()
