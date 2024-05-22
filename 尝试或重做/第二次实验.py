import numpy
import numpy as np
import matplotlib.pyplot as plt


x_train = numpy.array([8., 3., 9., 7., 16., 05., 3., 10., 4., 6.])
y_train = numpy.array([0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
x_test = numpy.array([5., 4.5, 9.8, 8., 22., 17., 3., 19., 20, 30])
y_test = numpy.array([0, 0, 0, 1, 1, 1, 0, 1, 1, 1])

weight_array = []
bias_array = []
loss_array = []

weight = 1
bias = 1

epochs = 2000


def Linear(X, w, b):
    return w * X + b


def clear():
    weight_array.clear()
    bias_array.clear()
    loss_array.clear()


def append(loss, weight, bias):
    loss_array.append(loss)
    weight_array.append(weight)
    bias_array.append(bias)


def Sigmoid(X, w, b):
    return numpy.exp( Linear(X, w, b)) / (1 + numpy.exp( Linear(X, w, b)))


def loss(X, y, w, b):
    return sum(-(y * numpy.log(Sigmoid(X, w, b)) + (1 - y) * numpy.log(1 - Sigmoid(X, w, b))))


def gradient(X, y, w, b):
    m = len(y)
    y_pred = Sigmoid(X, w, b)
    dw = sum(X * (y_pred - y)) / m
    db = np.mean(y_pred - y)
    return dw, db


def accuracy(y_hat, y):
    compare = y == y_hat
    return float(sum(compare)) / float(len(compare))


def precision(y_hat, y):
    cnt = 0
    sum = 0
    for i in range(len(y)):
        if y_hat[i] == 1:
            sum += 1
            if y[i] == 1:
                cnt += 1
    return float(cnt) / float(sum)


def recall(y_hat, y):
    return float(sum(y_hat)) / float(sum(y))


def roc(y_hat, y):
    thresholds = sorted(set(y_hat), reverse=True)
    tpr = []
    fpr = []
    for threshold in thresholds:
        y_pred = (y_hat >= threshold)
        true_positive = sum((y_pred == 1) & (y == 1))
        false_positive = sum((y_pred == 1) & (y == 0))
        true_negative = sum((y_pred == 0) & (y == 0))
        false_negative = sum((y_pred == 0) & (y == 1))
        tpr.append(true_positive / (true_positive + false_negative))
        fpr.append(false_positive / (false_positive + true_negative))
    return fpr, tpr


def SGD(X, y, w, b, learning_rate):
    clear()
    append(loss(X, y, w, b), w, b)
    for i in range(epochs):
        l_w, l_b = gradient(X, y, w, b)
        w -= learning_rate * l_w
        b -= learning_rate * l_b
        append(loss(X, y, w, b), w, b)


def Adagrad(X, y, w, b, learning_rate):
    clear()
    append(loss(X, y, w, b), w, b)
    phi = 1e-6
    sigma_w = 0
    sigma_b = 0
    for i in range(epochs):
        l_w, l_b = gradient(X, y, w, b)
        sigma_w = sigma_w+l_w * l_w
        sigma_b = sigma_b+l_b * l_b
        w -= learning_rate * l_w / (pow(sigma_w, 0.5) + phi)
        b -= learning_rate * l_b / (pow(sigma_b, 0.5) + phi)
        append(loss(X, y, w, b), w, b)


def Momentum(X, y, w, b, learning_rate, momentum=0.9):
    clear()
    append(loss(X, y, w, b), w, b)
    grad_w = 0
    grad_b = 0
    for i in range(epochs):
        l_w, l_b = gradient(X, y, w, b)
        grad_w = momentum * grad_w + (1 - momentum) * l_w
        grad_b = momentum * grad_b + (1 - momentum) * l_b
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b
        append(loss(X, y, w, b), w, b)


def RMSprop(X, y, w, b, learning_rate, p=0.9):
    clear()
    append(loss(X, y, w, b), w, b)
    phi = 1e-6
    sigma_w = 0
    sigma_b = 0
    for i in range(epochs):
        l_w, l_b = gradient(X, y, w, b)
        sigma_w = p * sigma_w + (1 - p) * l_w * l_w
        sigma_b = p * sigma_b + (1 - p) * l_b * l_b
        w -= learning_rate * l_w / (pow(sigma_w, 0.5) + phi)
        b -= learning_rate * l_b / (pow(sigma_b, 0.5) + phi)
        append(loss(X, y, w, b), w, b)


def plot_fig(X, y_hat,y):
    fig, ax = plt.subplots(1,2, figsize=(10,10))
    w_range = np.linspace(-10, 2, 1000)
    b_range = np.linspace(-50, 10, 1000)
    W, B = np.meshgrid(w_range, b_range)
    Z = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            Z[j, i] = loss(X, y, W[j, i], B[j, i])  # 假设你已经有了数据集 X 和标签 y，以及代价函数 loss 的定义

    fpr,tpr=roc(y_hat,y)
    ax[0].plot(fpr, tpr)
    ax[1].contourf(W, B, Z)
    ax[1].set_xlabel('w')
    ax[1].set_ylabel('b')
    ax[1].plot(weight_array, bias_array)

    plt.show()
    # return ax


SGD(x_train, y_train, weight, bias, 0.1)
weight = weight_array[epochs - 1]
bias = bias_array[epochs - 1]
y_hat = Sigmoid(x_test, weight, bias)
print('SGD')

print("Accuracy: {:.2f}".format(accuracy(y_hat>0.5, y_test)))
print("Precision: {:.2f}".format(precision(y_hat>0.5, y_test)))
print("Recall: {:.2f}".format(recall(y_hat>0.5, y_test)))

print("预测概率值：",y_hat)
y_np=np.array(y_hat>0.5)
print("预测分类：",y_np.astype(int))


plot_fig(x_train,y_hat,y_train)

step = [i for i in range(epochs + 1)]

plt.plot(step, loss_array,marker='*')

weight=1
bias=1
Adagrad(x_train, y_train, weight, bias, 0.5)
weight = weight_array[epochs - 1]
bias = bias_array[epochs - 1]
y_hat = Sigmoid(x_test, weight, bias)
plt.plot(step, loss_array,marker='*' )

print('ADAGRAD')

print("Accuracy: {:.2f}".format(accuracy(y_hat>0.5, y_test)))
print("Precision: {:.2f}".format(precision(y_hat>0.5, y_test)))
print("Recall: {:.2f}".format(recall(y_hat>0.5, y_test)))


weight=1
bias=1
RMSprop(x_train,y_train,weight,bias,0.03)

weight = weight_array[epochs - 1]
bias = bias_array[epochs - 1]
y_hat = Sigmoid(x_test, weight, bias)
print('RMSPROP')

print("Accuracy: {:.2f}".format(accuracy(y_hat>0.5, y_test)))
print("Precision: {:.2f}".format(precision(y_hat>0.5, y_test)))
print("Recall: {:.2f}".format(recall(y_hat>0.5, y_test)))
plt.plot(step, loss_array,marker='*')
weight=1
bias=1
Momentum(x_train,y_train,weight,bias,0.2)

weight = weight_array[epochs - 1]
bias = bias_array[epochs - 1]
y_hat = Sigmoid(x_test, weight, bias)
print('Momentum')

print("Accuracy: {:.2f}".format(accuracy(y_hat>0.5, y_test)))
print("Precision: {:.2f}".format(precision(y_hat>0.5, y_test)))
print("Recall: {:.2f}".format(recall(y_hat>0.5, y_test)))
plt.plot(step, loss_array,marker='*')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['SGD', 'Adagrad', 'Momentum', 'RMSprop'])
plt.show()

