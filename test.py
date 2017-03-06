import nn
import utils
import numpy as np
import pandas as pd
import random
import itertools

def check_nn_gen_delault_connection():
    n = nn.unit(2, 2, 2)
    n.gen_default_connection(10, 0)
    print(n.connection)


def check_clone(fp):
    ml = nn.link(0, 0)
    ml.clone(fp)
    ml.describe()


def check_get_latest():
    ml = nn.link(0, 0)
    ml.get_latest()


def check_ematrix():
    print(utils.gen_ematrix(5))


def check_gen_id(n):
    print(utils.gen_id(n))


def anonymous():
    print(a)
    exit()
    a = np.array([[10, 20, 30], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    b = np.array([[1, 2, 3, 4], [10, 10, 10, 10]])
    # f_ = lambda x: np.random.normal(x, 10)
    # b = np.frompyfunc([f_, f_, f_, f_, f_])
    # print(b * a)
    # print(1 / np.sqrt(b))
    # print(a.std())
    exit()
    b = 5
    a = [1, 2, b, 4, 5]
    for i, j in zip(a, a[1:]):
        print(i, j)
        b = 100
    # a = lambda a, b: a + b
    # print(a(*(1, 2)))


if __name__ == '__main__':
    # check_get_latest()
    # check_clone("./pickle/gen0_score0p4989_2017_0303_140511")
    # check_nn_gen_delault_connection()
    # check_gen_id(2)
    anonymous()
    # demo_singl_unit()
    # demo_linked()
    # check_ematrix()
