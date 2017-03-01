import numpy as np


class fanction(object):
    def __init__(self, fanc, derv, name):
        self.fanc = fanc
        self.derv = derv
        self.name = name


def f_tanh(x):
    return np.tanh(x)


def f_dtanh(y):
    """d (tanh(x)) / dx = 1-tanh2(x)
    """
    return 1. - y**2

tanh = fanction(f_tanh, f_dtanh, "tanh")


def f_relu(x):
    return max(0, x)


def f_drelu(y):
    if y >= 0: return 1
    else: return 0

relu = fanction(f_relu, f_drelu, "relu")
