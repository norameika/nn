import nn
import pandas as pd
import numpy as np
import utils


def pat_train(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(40000)[5000::2], nrows=5000)
    res = list()
    for l in df.values:
        res.append([np.array([(i - mean) / sig for i in l[1:]]), np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


def pat_eval(fp):
    mean, sig = 33.372029299363057, 78.634459506389177
    n_out = 10
    df = pd.read_csv(fp, iterator=True, skiprows=range(10)[1::2], nrows=100)
    res = list()
    for l in df.values:
        res.append([np.array([(i - mean) / sig for i in l[1:]]), np.array([1 if i == l[0] else 0 for i in range(n_out)])])
    return res


class mylink(nn.link):
    def __init__(self, n_input, n_output):
        nn.link.__init__(self, n_input, n_output)

    def design(self):
        self.interposer_input = nn.interposer(self.n_input)
        self.interposer_interm = nn.interposer(300)
        self.interposer_output = nn.interposer(10)
        self.interposers = [self.interposer_input, self.interposer_interm, self.interposer_output]

        # design 1st layer
        unit00 = nn.unit(self.n_input, self.n_input, 300)
        unit00.name = "00"
        unit00.gen_default_connection(self.n_input, 0)
        self.layers.append([unit00])

        # # design 2nd layer
        unit10 = nn.unit(300, 200, 10)
        unit10.gen_default_connection(300, 0)
        unit10.name = "10"
        unit10.const = 0
        self.layers.append([unit10])

    def evaluator(self, res, tar):
        print(pd.DataFrame({"res": res, "tar": tar}))
        print("reslut, target -> %s, %s" % (list(res).index(max(res)), list(tar).index(1)))
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


def mnist(fp):
    # create a network with two input, two hidden, and one output nodes
    ml = mylink(784, 10)
    ml.get_latest()

    # a = utils.animator()
    # pat = pat_train(fp)
    pat = pat_train(fp)
    print("start training for %s x %s datasets" % (pat[0][0].shape, len(pat)))
    # for d in ml.train(res, epoch=50000):
    #     pass
    # n = nn.unit(784, 100, 10)
    # n.set_pattern(pat)
    # n.initialization("random")
    # for i in ml.train(pat, epoch=100):
    #     pass
    ani = utils.animator()
    ani.arrange_for_animation(ml.train(pat, epoch=100, save=1, interval=1))
    ani.animation()
    # a.arrange_for_animation(ml.train(pat_train(fp), epoch=50000))
    # a.animation()
    ml.evaluate(pat_eval(fp))

if __name__ == '__main__':
     mnist("./mnist/train.csv")


