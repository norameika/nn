import nn
import utils
import numpy as np
import pandas as pd


def check_nn_gen_delault_connection():
    n = nn.unit(2, 2, 2)
    n.gen_default_connection(10, 0)
    print(n.connection)


def check_clone(fp):
    ml = nn.link(0, 0)
    ml.clone(fp)
    ml.describe()


def check_get_latest():
    ml = nn.link(0, 0)
    ml.get_latest()


def demo_singl_unit():
    pat = [
        [[0, 0, 0], [0]],
        [[0, 0, 1], [1]],
        [[0, 1, 0], [1]],
        [[0, 1, 1], [0]],
        [[1, 0, 0], [1]],
        [[1, 0, 1], [0]],
        [[1, 1, 0], [0]],
        [[1, 1, 1], [0]],
    ]

    # create a network with two input, two hidden, and one output nodes
    n = nn.unit(3, 10, 1)
    n.set_pattern(pat)
    # n.initialization("random")
    ani = utils.animator()
    ani.arrange_for_animation(n.train(pat, epoch=10000))
    ani.animation()
    n.evaluate(pat)


class mylink(nn.link):
    def __init__(self, n_input, n_output):
        nn.link.__init__(self, n_input, n_output)

    def design(self):
        self.interposer_input = nn.interposer(self.n_input)
        self.interposer_interm = nn.interposer(4)
        self.interposer_output = nn.interposer(1)
        self.interposers = [self.interposer_input, self.interposer_interm, self.interposer_output]

        # design 1st layer
        unit00 = nn.unit(3, 4, 4)
        unit00.name = "00"
        unit00.gen_default_connection(self.n_input, 0)
        self.layers.append([unit00])

        # # design 2nd layer
        unit10 = nn.unit(4, 3, 1)
        unit10.gen_default_connection(4, 0)
        unit10.name = "10"
        unit10.const = 0
        self.layers.append([unit10])

    def evaluator(self, res, tar):
        print(pd.DataFrame({"res": res,  "tar": tar}))
        print("reslut, target -> %s, %s" % (res[0], tar[0]))
        if round(res[0]) == tar[0]:
            return 1
        else:
            return 0


def demo_linked():
    pat = [
        [[0, 0, 0], [0]],
        [[0, 0, 1], [1]],
        [[0, 1, 0], [1]],
        [[0, 1, 1], [0]],
        [[1, 0, 0], [1]],
        [[1, 0, 1], [0]],
        [[1, 1, 0], [0]],
        [[1, 1, 1], [0]],
    ]

    # create a network with two input, two hidden, and one output nodes
    ml = mylink(3, 1)

    a = utils.animator()
    a.arrange_for_animation(ml.train(pat, epoch=10000))
    a.animation()
    ml.evaluate(pat)


def anonymous():
    a = np.array([[1, 0], [0, 2]])
    print(a.std())
    exit()
    b = 5
    a = [1, 2, b, 4, 5]
    for i, j in zip(a, a[1:]):
        print(i, j)
        b = 100
    # a = lambda a, b: a + b
    # print(a(*(1, 2)))


def check_ematrix():
    print(utils.gen_ematrix(5))

if __name__ == '__main__':
    # check_get_latest()
    # check_clone("./pickle/gen0_score0p4989_2017_0303_140511")
    # check_nn_gen_delault_connection()
    # anonymous()
    # demo_singl_unit()
    demo_linked()
    # check_ematrix()
