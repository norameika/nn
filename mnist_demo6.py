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

    # pat = pat_train(fp, 0, 120000)
    # print(len(pat))
    # exit()

    # with open('./pickle/pat_train', mode='wb') as f:
    #     pickle.dump(pat, f)
    # exit()
    while 1:
        u = myunit(784*2, 2000, 10)
        u.activation_temp = 0.01
        # u.alpha = 6.152480735753703e-05
        # u.beta = 0.8693087894881881
        # u.gamma = 0.8111022047365893
        # u.name = "serval_%s" % utils.gen_id(2)
        # u.name = "okaping"
        # u.initialization("gaussian", 0, u.activation_temp)
        u.clone(get_pickle("usagifft_fo"))
        u.delta = 0.33
        sindex = random.randint(0, 40000*1 - 40000)
        pat = pat_train(fp, sindex, 40000, 1)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, u.evaluate, (pat_eval(fp), 1), epoch=100, interval=1): pass

    #bkm for 748 * 10
    # u.activation_temp = 0.0001
    # u.initialization("gaussian", 0, u.activation_temp)
    # u.alpha = 0.00001
    # u.beta = 0.9
    # u.gamma = 0.9


if __name__ == '__main__':
    mnist("./mnist/train.csv")
