import nn
import utils
import numpy as np
import pandas as pd
import random
import itertools
import functions


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

def check_convolve():
    u = nn.unit(1, 1, 1)
    arr = np.array([1, 2, 3, 4, 10, 20, 30, 40, 1, 2, 3, 4, 10, 20, 30, 40,])
    print(u.convolve(arr, masksize=1).mean())


def anonymous():
    a = [1, 2, 4, 6]
    random.shuffle(a)
    print(a)
    print(utils.gen_mask_fft(10, 8))
    # print(np.matmul(a, b))
if __name__ == '__main__':
    # check_get_latest()
    # check_clone("./pickle/gen0_score0p4989_2017_0303_140511")
    # check_nn_gen_delault_connection()
    # check_gen_id(2)
    check_convolve()
    # anonymous()
    # demo_singl_unit()
    # demo_linked()
    # check_ematrix()
