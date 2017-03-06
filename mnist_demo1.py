import nn
import pandas as pd
import numpy as np
import utils
import itertools
import functions
import random


class myunit(nn.unit):
    def __init__(self, *args):
        nn.unit.__init__(self, *args)

    def evaluator(self, res, tar):
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


u0 = myunit(1, 1, 1)
u0.clone("./pickle/searabbit_s2_mrty_gen1_score94p95_2017_0304_222940")
u1 = myunit(1, 1, 1)
u1.clone("./pickle/searabbit_s3_ay_gen0_score81p8_2017_0304_212840")


def pat_train(fp, n, m):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([np.append(u0.forward_propagation(inputs), u1.forward_propagation(inputs)), np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_eval(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000), nrows=2000)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([np.append(u0.forward_propagation(inputs), u1.forward_propagation(inputs)), np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def mnist(fp):
    # create a network with two input, two hidden, and one output nodes
    def add_newer(gen):
        layer = np.random.randint(10, 30)
        u = myunit(20, layer, 10)
        u.initialization("gaussian", 0, 0.01 * 100. / float(u.n_layers[1]))
        u.alpha = abs(np.random.normal(0., 0.0001 * 100. / float(u.n_layers[1])))
        u.name = "whale_s%s_%s" % (gen, utils.gen_id(2))
        return u

    def add_child(g):
        res = list()
        for u0, u1 in itertools.combinations(g, 2):
            if np.random.normal(0, 1) <= 0:
                n_layers_new = [max(i, j) for i, j in zip(u0.n_layers, u1.n_layers)]
            else:
                n_layers_new = [min(i, j) for i, j in zip(u0.n_layers, u1.n_layers)]
            n_layers_new = (np.array(n_layers_new) - np.array([1, 0, 0])).tolist()
            child = myunit(*n_layers_new)
            child.name = "whale_s%s_%s" % (gen, u0.name.split("_")[-1]+u1.name.split("_")[-1])
            res.append(u0.reproduce(u1, child))
        return res

    group = list()
    survier = 3

    """initialize"""
    for i in range(5):
        group.append(add_newer(0))
    sindex = random.randint(0, 2000 * (int(40000 / 2000.) - 1))
    pat = pat_train(fp, sindex, 2000)
    for u in group:
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, epoch=20, interval=1): pass
        u.evaluate(pat_eval(fp), save=1)
    group = sorted(group, key=lambda x: x.score, reverse=1)
    print("genration%s, " % 0, ", ".join(["%-.2f" % x.score for x in group]))
    group[:survier]
    """"""""""""""""""

    for gen in range(1, 10):
        sindex = random.randint(0, 30000)
        pat = pat_train(fp, sindex, 10000)
        group = group[:survier-1]
        group.append(add_newer(gen))
        group += add_child(group)
        for u in group:
            print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
            u.describe()
            for _ in u.train(pat, epoch=25, interval=1): pass
            u.evaluate(pat_eval(fp), save=1)

        group = sorted(group, key=lambda x: x.score, reverse=1)
        print("genration%s, " % gen, ", ".join(["%-.2f" % x.score for x in group]))
        group[:survier]


if __name__ == '__main__':
    mnist("./mnist/train.csv")
