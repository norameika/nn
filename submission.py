import nn
import pandas as pd
import numpy as np
import utils
import itertools
import random
import functions
import os
import pickle
from PIL import Image


class myunit(nn.unit):
    def __init__(self, *args):
        nn.unit.__init__(self, *args)

    def evaluator(self, res, tar):
        if res.max() != res.max(): return 0
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


def pat_train(fp, n, m, multiply=3):
    try:
        with open("./pickle/pat_train_%s" % multiply, mode='rb') as f:
            d = pickle.load(f)
            random.shuffle(d)
            return d[n:n + m]
    except:
        pass


def pat_eval(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000), nrows=2000)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([inputs, np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_exam(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    df = pd.read_csv(fp)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l])
        res.append([inputs])
    return res


def mnist(fp):
    def get_pickle(name):
        res = list()
        for f in os.listdir("./pickle"):
            if name in f: res.append(f)
        res = sorted(res, key=lambda x: os.path.getmtime("./pickle/%s" % x), reverse=1)[0]
        return "./pickle/%s" % res
    u = myunit(784, 1000, 10)
    u.clone(get_pickle("capibara_kx"))
    pat = pat_exam(fp)
    res = list()
    for cnt, p in enumerate(pat):
        out = u.forward_propagation(p)
        # print (out, list(out).index(max(out)))
        res.append([cnt, list(out).index(max(out))])
    pd.DataFrame(res, columns=["ImageId", "Label"]).to_csv("submission.csv")

    #bkm for 748 * 10
    # u.activation_temp = 0.0001
    # u.initialization("gaussian", 0, u.activation_temp)
    # u.alpha = 0.00001
    # u.beta = 0.9
    # u.gamma = 0.9


if __name__ == '__main__':
    mnist("./mnist/test.csv")
