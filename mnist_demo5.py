import nn
import pandas as pd
import numpy as np
import utils
import itertools
import random
import functions
import os
import pickle


class myunit(nn.unit):
    def __init__(self, *args):
        nn.unit.__init__(self, *args)

    def evaluator(self, res, tar):
        if res.max() != res.max(): return 0
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


def pat_train(fp, n, m):
    try:
        with open("./pickle/pat_train", mode='rb') as f:
            d = pickle.load(f)
            return d[n:n + m]
    except:
        pass
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([inputs, np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_eval(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000), nrows=2000)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([inputs, np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def mnist(fp):
    def get_pickle(name):
        res = list()
        for f in os.listdir("./pickle"):
            if name in f: res.append(f)
        res = sorted(res, key=lambda x: os.path.getmtime("./pickle/%s" % x), reverse=1)[0]
        return "./pickle/%s" % res

    while 1:
        u = myunit(784, 800, 800, 10)
        # u.activation_temp = 0.005239828201117102
        # u.alpha = 3.6349203390986874e-05
        # u.beta = 0.7286449556280377
        # u.gamma = 0.7327171199235663
        # u.name = "serval_%s" % utils.gen_id(2)
        # u.initialization("gaussian", 0, u.activation_temp)
        u.delta = 0.1
        u.clone(get_pickle("usagi_fn"))
        u.alpha = 5.136056622452953e-05
        sindex = random.randint(0, 30000)
        pat = pat_train(fp, 0, 40000)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, u.evaluate, (pat_eval(fp), 1), epoch=1, interval=1): pass

    #bkm for 748 * 10
    # u.activation_temp = 0.0001
    # u.initialization("gaussian", 0, u.activation_temp)
    # u.alpha = 0.00001
    # u.beta = 0.9
    # u.gamma = 0.9


if __name__ == '__main__':
    mnist("./mnist/train.csv")
