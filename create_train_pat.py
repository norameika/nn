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


def pat_train(fp, n, m):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        res.append([inputs, np.array([1 if i == l[0] else 0 for i in range(n_out)])])
        for i in range(4):
            rot = min(10, max(-10, np.random.normal(0, 2)))
            # zoom = min(1.1, max(0.9, np.random.normal(1, 0.02)))
            img = Image.fromarray(np.uint8(l[1:]).reshape(28, 28))
            img = img.rotate(rot)
            # img = utils.crop_image(img, (0.5, 0.5), zoom)
            inputs = np.array(img).reshape(28 * 28)
            inputs = np.array([(i - mean) / sig for i in inputs])
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

    pat = pat_train(fp, 0, 40000)
    with open('./pickle/pat_train_5', mode='wb') as f:
        pickle.dump(pat, f)
    exit()

if __name__ == '__main__':
    mnist("./mnist/train.csv")