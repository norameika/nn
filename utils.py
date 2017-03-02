import numpy as np
import random


def rand(a, b):
    return (b - a) * random.random() + a


def gen_matrix(i, j, fill=0.):
    matrix = []
    for i in range(i):
        matrix.append([fill] * j)
    return np.array(matrix)


def gen_ematrix(n, inp=0):
    matrix = []
    for i in range(n):
        matrix.append([0 if k != i else 1 for k in range(n + inp)])
    return np.array(matrix)
