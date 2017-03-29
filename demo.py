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
    for i in range(500): res += pat
    random.shuffle(res)
    return res


def pat_eval():
    pat = [
        [[[0, 0, 0]], [0, 1]],
        [[[0, 0, 1]], [1, 0]],
        [[[0, 1, 0]], [1, 0]],
        [[[0, 1, 1]], [0, 1]],
        [[[1, 0, 0]], [0, 1]],
        [[[1, 0, 1]], [1, 0]],
        [[[1, 1, 0]], [1, 0]],
        [[[1, 1, 1]], [0, 1]],
    ]
    res = list()
    for i in range(20): res += pat
    random.shuffle(res)
    return res

import nn


def demo():
    u = nn.unit(3, 200, 2)
    u.name = "test"
    u.comment = "demo for XOR"
    u.set_activation_func([functions.relu, functions.tanh])
    u.cost_func = functions.logloss
    for _ in u.train(pat_train(), u.evaluate, (pat_eval(), 0), epoch=2, pre_unit_train=0): pass

if __name__ == '__main__':
    demo()
