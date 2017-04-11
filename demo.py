import random
import functions
import utils


def pat_train():
    pat = [
        [[[0, 0, 0], ], [0, 1]],
        [[[0, 0, 1], ], [1, 0]],
        [[[0, 1, 0], ], [1, 0]],
        [[[0, 1, 1], ], [0, 1]],
        [[[1, 0, 0], ], [0, 1]],
        [[[1, 0, 1], ], [1, 0]],
        [[[1, 1, 0], ], [1, 0]],
        [[[1, 1, 1], ], [0, 1]],
    ]
    res = list()
    for i in range(1000): res += pat
    random.shuffle(res)
    return res


def pat_eval():
    pat = [
        [[[0, 0, 0], ], [0, 1]],
        [[[0, 0, 1], ], [1, 0]],
        [[[0, 1, 0], ], [1, 0]],
        [[[0, 1, 1], ], [0, 1]],
        [[[1, 0, 0], ], [0, 1]],
        [[[1, 0, 1], ], [1, 0]],
        [[[1, 1, 0], ], [1, 0]],
        [[[1, 1, 1], ], [0, 1]],
    ]
    res = list()
    for i in range(20): res += pat
    random.shuffle(res)
    return res

import nn


def demo():
    mymodel = nn.model()
    mymodel.name = "test"
    mymodel.add_propable([nn.fc(3, 100)])
    mymodel.add_propable([nn.node(100)])
    mymodel.add_propable([nn.fc(100, 2)])
    mymodel.add_propable([nn.node_out(2)])

    mymodel.train(pat_train(), 10)
    mymodel.eval(pat_eval())

if __name__ == '__main__':
    demo()