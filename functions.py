import numpy as np


class function(object):
    def __init__(self, func, derv, name):
        self.func = func
        self.derv = derv
        self.name = name


def f_tanh(x):
    return np.tanh(x)


def f_dtanh(y):
    """d (tanh(x)) / dx = 1-tanh2(x)
    """
    return 1. - y**2

tanh = function(f_tanh, f_dtanh, "tanh")


def f_relu(x):
    return max(0, x)


def f_drelu(y):
    if y >= 0: return 1
    else: return 0

relu = function(f_relu, f_drelu, "relu")


def f_square_error(output, targets):
    return sum([(i - j) ** 2 for i, j in zip(output, targets)])


def f_de_square_error(output, targets):
    return targets - output

square_error = function(f_square_error, f_de_square_error, "square_error")
