import numpy as np


class activation_func(object):
    def __init__(self, func, derv, name):
        self.func = func
        self.derv = derv
        self.name = name


class cost_func(object):
    def __init__(self, func, derv, cost, name):
        self.func = func
        self.derv = derv
        self.cost = cost
        self.name = name


def f_tanh(x):
    return np.tanh(x)


def f_dtanh(y):
    """d (tanh(x)) / dx = 1-tanh2(x)
    """
    return 1. - y ** 2

tanh = activation_func(f_tanh, f_dtanh, "tanh")


def f_relu(x):
    return max(0, x)


def f_drelu(y):
    if y >= 0: return 1
    else: return 0

relu = activation_func(f_relu, f_drelu, "relu")


def f_square_error(output):
    return output


def f_square_cost(output, targets):
    return sum([(i - j) ** 2 for i, j in zip(output, targets)]) / 2


def f_de_square_error(output, targets):
    return targets - output

square_error = cost_func(f_square_error, f_de_square_error, f_square_cost, "square_error")


def f_softmax(output):
    return np.exp(output) / sum(np.exp(output))


def f_softmax_cost(output, targets):
    output = f_softmax(output)
    return - sum([j * np.log(i) for i, j in zip(output, targets)])


def f_de_softmax(output, targets):
    output = f_softmax(output)
    return (targets - output)
softmax = cost_func(f_softmax, f_de_softmax, f_softmax_cost, "softmax")


# def f_logsoftmax(output):
#     return output - np.log(sum(np.exp(output)))


# def f_logsoftmax_cost(output, targets):
#     output = f_softmax(output)
#     return - sum([j * np.log(i) for i, j in zip(output, targets)])


# def f_de_logsoftmax(output, targets):
#     output = f_logsoftmax(output)
#     return (targets - output)

# logsoftmax = cost_func(f_logsoftmax, f_de_logsoftmax, f_logsoftmax_cost, "logsoftmax")