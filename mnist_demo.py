import nn
import pandas as pd
import numpy as np


def pat_train(fp="data/train.csv", n=0, m=10000):
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inp = np.array(l[1:])
        inp = (inp - min(inp))/(max(inp) - min(inp))
        res.append([[inp, ], np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_eval(fp="data/train.csv", n=40000, m=1000):
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[:n], nrows=m)
    res = list()
    for l in df.values:
        inp = np.array(l[1:])
        inp = (inp - min(inp))/(max(inp) - min(inp))
        res.append([[inp, ], np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def demo():
    mymodel = nn.model()
    mymodel.name = "test"
    mymodel.add_prop([nn.fc(28 * 28, 100)])
    mymodel.add_prop([nn.node(100)])
    mymodel.add_prop([nn.fc(100, 10)])
    mymodel.add_prop([nn.node_out(10)])

    mymodel.train(pat_train(), pat_eval(), 10)

if __name__ == '__main__':
    demo()
