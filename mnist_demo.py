import nn.nn
import nn.functions
import pandas as pd
import numpy
import pickle
import os
import random

# C:/Users/yyonai/python34/Scripts/python.exe



def pat_train(fp, n, m):
    try:
        with open("./pickle/pat_train", mode='rb') as f:
            d = pickle.load(f)
            return d[n: n + m]
    except:
        pass
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inputs = np.array([(i - mean) / sig for i in l[1:]])
        inputs = (inputs - min(inputs))/(max(inputs) - min(inputs))
        res.append([inputs, np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_eval(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000), nrows=2000)
    res = list()
    for l in df.values:
        inputs = numpy.array([(i - mean) / sig for i in l[1:]])
        inputs = (inputs - min(inputs))/(max(inputs) - min(inputs))
        res.append([inputs, numpy.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def save_pat(fp):
    pat = pat_train(fp, 0, 40000)
    with open("./pickle/pat_train", mode="wb") as f:
        pickle.dump(pat, f)


def get_pickle(name):
    res = list()
    for f in os.listdir("./pickle"):
        if name in f: res.append(f)
    res = sorted(res, key=lambda x: os.path.getmtime("./pickle/%s" % x), reverse=1)[0]
    return "./pickle/%s" % res


def mnist(fp):
    while 1:
        pat = [[[numpy.array(i[0]).reshape(28, 28), ], i[1]] for i in pat_train(fp, 0, 10000)]
        for i in range(10): random.shuffle(pat)
        pat_e = [[[numpy.array(i[0]).reshape(28, 28), ], i[1]] for i in pat_eval(fp)]
        u = nn.nn.model()
        u.name = nn.utils.gen_id(4)
        kn = 10
        u.add_prop([nn.nn.cnvl(5, 1, (28, 28), kn)])
        u.add_prop([nn.nn.maxpool(2, 2, (24, 24)) for _ in range(kn)])
        u.add_prop([nn.nn.deploy_nested(12 * 12 * kn)])
        u.add_prop([nn.nn.cnvl(5, 1, (12, 12), 2, zp=2) for _ in range(kn)])
        u.add_prop([nn.nn.deploy_nested(12 * 12 * kn * 2)])
        u.add_prop([nn.nn.cnvl(5, 1, (12, 12), 2, zp=2) for _ in range(kn * 2)])
        u.add_prop([nn.nn.maxpool(2, 2, (12, 12)) for _ in range(kn * 2 * 2)])
        u.add_prop([nn.nn.deploy()])
        u.add_prop([nn.nn.fc(6 * 6 * kn * 2 * 2, 10)])
        u.add_prop([nn.nn.node_out(10)])
        u.set_params()
        u.train(pat, pat_e, 10, freq=100)

if __name__ == '__main__':
    mnist("./mnist/train.csv")
