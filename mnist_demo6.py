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


def mnist(fp):
    def get_pickle(name):
        res = list()
        for f in os.listdir("./pickle"):
            if name in f: res.append(f)
        res = sorted(res, key=lambda x: os.path.getmtime("./pickle/%s" % x), reverse=1)[0]
        return "./pickle/%s" % res
    u = myunit(28*28, 1000, 10)
    u.clone(get_pickle("capibara_kx"))
    u.alpha *= 0.2
    while 1:
        sindex = random.randint(0, 40000 - 40000)
        pat = pat_train(fp, sindex, 40000)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, u.evaluate, (pat_eval(fp), 1), epoch=100, interval=1): pass

if __name__ == '__main__':
    mnist("./mnist/train.csv")
