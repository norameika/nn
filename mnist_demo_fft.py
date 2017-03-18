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
        with open("./pickle/pat_train_fft", mode='rb') as f:
            d = pickle.load(f)
            random.shuffle(d)
            return d[n:n + m]
    except:
        pass
    mean, sig = 33.372029299363057, 78.634459506389177
    mean_fft, sig_fft = 160.427483654, 27.5728007433
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inputs_fft = np.fft.fft2(np.array(l[1:]).reshape(28, 28))
        inputs_fft = np.log(np.abs(inputs_fft) + 1)
        inputs_fft = inputs_fft / np.amax(inputs_fft) * 255
        inputs_fft = inputs_fft.reshape(784, 1)
        inputs_fft = np.array([(i - mean_fft) / sig_fft for i in inputs_fft])
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([np.append(inputs, inputs_fft), np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_eval(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    mean_fft, sig_fft = 160.427483654, 27.5728007433
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000), nrows=2000)
    res = list()
    for l in df.values:
        inputs_fft = np.fft.fft2(np.array(l[1:]).reshape(28, 28))
        inputs_fft = np.log(np.abs(inputs_fft) + 1)
        inputs_fft = inputs_fft / np.amax(inputs_fft) * 255
        inputs_fft = inputs_fft.reshape(784, 1)
        inputs_fft = np.array([(i - mean_fft) / sig_fft for i in inputs_fft])
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([np.append(inputs, inputs_fft), np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def get_pickle(name):
    res = list()
    for f in os.listdir("./pickle"):
        if name in f: res.append(f)
    res = sorted(res, key=lambda x: os.path.getmtime("./pickle/%s" % x), reverse=1)[0]
    return "./pickle/%s" % res


def mnist_cherrypic(fp):
    while 1:
        u = myunit(784*2, 777, 777, 10)
        u.weights_mask[0] = utils.gen_mask_fft(*u.weights_mask[0].shape)
        u.activation_temp = 10 ** (-3 - np.random.normal(0, 1))
        u.alpha = 10 ** (-3 - np.random.normal(0, 1))
        u.beta = min(0.99, abs(np.random.normal(0.8, 0.05)))
        u.gamma = min(0.99, abs(np.random.normal(0.8, 0.05)))
        u.name = "usagifft_%s" % utils.gen_id(2)
        u.initialization("gaussian", 0, u.activation_temp)
        sindex = random.randint(0, 30000)
        pat = pat_train(fp, sindex, 10000)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, u.evaluate, (pat_eval(fp), 1), epoch=1, interval=1): pass


def mnist(fp):
    def get_pickle(name):
        res = list()
        for f in os.listdir("./pickle"):
            if name in f: res.append(f)
        res = sorted(res, key=lambda x: os.path.getmtime("./pickle/%s" % x), reverse=1)[0]
        return "./pickle/%s" % res

    for i in range(100):
        u = myunit(784*2, 000, 10)
        # u.activation_temp = 0.005239828201117102
        # u.alpha = 3.6349203390986874e-05
        # u.beta = 0.7286449556280377
        # u.gamma = 0.7327171199235663
        # u.name = "serval_%s" % utils.gen_id(2)
        # u.initialization("gaussian", 0, u.activation_temp)
        u.delta = 0.33
        u.clone(get_pickle("usagifft_fo"))
        # u.alpha = 5.329952064184424e-05
        sindex = random.randint(0, 30000)
        # sindex = 0
        pat = pat_train(fp, 0, 10000)
        print("%s start training for %s x %s datasets from %s" % (u.name, pat[0][0].shape, len(pat), sindex))
        u.describe()
        for _ in u.train(pat, u.evaluate, (pat_eval(fp), 1), epoch=3, interval=1): pass

    #bkm for 748 * 10
    # u.activation_temp = 0.0001
    # u.initialization("gaussian", 0, u.activation_temp)
    # u.alpha = 0.00001
    # u.beta = 0.9
    # u.gamma = 0.9


if __name__ == '__main__':
    # mnist("./mnist/train.csv")
    mnist_cherrypic("./mnist/train.csv")
