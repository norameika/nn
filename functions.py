import numpy as np
import numpy


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
    y = (y <= 1 and y >= -1) * y
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
    return ((output - targets) * (output - targets) / 2).sum()


def f_de_square_error(output, targets):
    return targets - output

square_error = cost_func(f_square_error, f_de_square_error, f_square_cost, "square_error")


def f_logloss(output):
    e_x = numpy.exp(output - numpy.max(output))
    return e_x / e_x.sum(axis=0)
    # return np.exp(output) / sum(np.exp(output))

cap = lambda x: max(min(x, 1 - 1E-15), 1E-15)
cap = numpy.vectorize(cap)


def f_logloss_cost(output, targets):
    output = cap(f_logloss(output))
    return - sum(targets * np.log(output))


def f_de_logloss(output, targets):
    output = f_logloss(output)
    return (output - targets)

logloss = cost_func(f_logloss, f_de_logloss, f_logloss_cost, "logloss")
