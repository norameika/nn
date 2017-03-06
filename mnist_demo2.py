import nn
import pandas as pd
import numpy as np
import utils
import itertools
import random
import functions


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


def mnist(fp):
    # create a network with two input, two hidden, and one output nodes
    def add_newer(gen):
        layer0 = np.random.randint(100, 800)
        layer1 = np.random.randint(12, layer0)
        layer2 = np.random.randint(11, layer1)
        # # layer3 = np.random.randint(10, layer2)
        u = myunit(784, 700, 200, 10)
        # a = abs(np.random.normal(0, 0.03))
        # print(a)
        # u.initialization("gaussian", 0, a)
        # u.alpha = np.random.normal(0.01, 0.001)
        # u.beta = min(0.99, np.random.normal(0.9, 0.09))
        # u.gamma = min(0.99, np.random.normal(0.9, 0.09))
        # u.cost_func = functions.softmax
        # u.set_activation_func([functions.relu])

        # bkm for sqerr
        u.initialization("gaussian", 0, 0.001)
        u.alpha = 0.00001
        u.beta = 0.9
        u.gamma = 0.9

        u.name = "cat_s%s_%s" % (gen, utils.gen_id(2))
        return u

    def add_child(g):
        res = list()
        for u0, u1 in itertools.combinations(g, 2):
            if np.random.normal(0, 1) <= 0:
                n_layers_new = [max(i, j) for i, j in zip(u0.n_layers, u1.n_layers)]
            else:
                n_layers_new = [min(i, j) for i, j in zip(u0.n_layers, u1.n_layers)]
            n_layers_new = (np.array(n_layers_new) - np.array([1, 0, 0, 0, 0])).tolist()
            child = myunit(*n_layers_new)
            child.name = "dog_s%s_%s" % (gen, u0.name.split("_")[-1]+u1.name.split("_")[-1])
            res.append(u0.reproduce(u1, child))
        return res

    group = list()
    survier = 3

    """initialize"""
    for i in range(1):
        group.append(add_newer(0))
    for u in group:
        sindex = 0
        pat = pat_train(fp, sindex, 40000)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        an = utils.animator()
        an.arrange_for_animation(u.train(pat, u.evaluate, (pat_eval(fp), 0), epoch=500, interval=1))
        an.animation()
        # for _ in u.train(pat, epoch=20, interval=1): pass
        u.evaluate(pat_eval(fp), save=1)
    group = sorted(group, key=lambda x: x.score, reverse=1)
    print("genration%s, " % 0, ", ".join(["%-.2f" % x.score for x in group]))

    exit()

    # """pickking"""
    # group = group[:2]
    # for u in group:
    #     sindex = random.randint(0, 3500)
    #     pat = pat_train(fp, sindex, 35000)
    #     print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
    #     u.describe()
    #     for _ in u.train(pat, epoch=50, interval=1): pass
    #     u.evaluate(pat_eval(fp), save=1)

    group[:survier]

    """ecocsyctem"""
    for gen in range(1, 10):
        sindex = random.randint(0, 3500)
        pat = pat_train(fp, sindex, 35000)
        group = group[:survier-1]
        # group.append(add_newer(gen))
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
