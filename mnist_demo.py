import nn
import pandas as pd
import numpy as np
import utils
import itertools
import random


class myunit(nn.unit):
    def __init__(self, *args):
        nn.unit.__init__(self, *args)

    def evaluator(self, res, tar):
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


def pat_train(fp, n, m):
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


    group = list()
    survier = 2

    """initialize"""
    for i in range(1):
        group.append(add_newer(0))
    for u in group:
        sindex = random.randint(0, 2000 * (int(40000 / 2000.) - 1))
        pat = pat_train(fp, sindex, 2000)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, epoch=5, interval=1): pass
        u.evaluate(pat_eval(fp), save=1)
    group = sorted(group, key=lambda x: x.score, reverse=1)
    print("genration%s, " % 0, ", ".join(["%-.2f" % x.score for x in group]))

    exit()

    """ecocsyctem"""
    for gen in range(1, 10):
        sindex = random.randint(0, 17000)
        pat = pat_train(fp, sindex, 20000)
        group = group[:survier]
        group.append(add_newer(gen))
        group += add_child(group)
        for u in group:
            print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
            u.describe()
            for _ in u.train(pat, epoch=20, interval=1): pass
            u.evaluate(pat_eval(fp), save=1)

        group = sorted(group, key=lambda x: x.score, reverse=1)
        print("genration%s, " % gen, ", ".join(["%-.2f" % x.score for x in group]))
        group[:survier]


if __name__ == '__main__':
    mnist("./mnist/train.csv")
