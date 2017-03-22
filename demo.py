import random
import functions


def pat_train():
    pat = [
        [[0, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 0], [1, 0]],
        [[1, 1], [0, 1]],
    ]
    res = list()
    for i in range(5000): res += pat
    random.shuffle(res)
    return res


def pat_eval():
    pat = [
        [[0, 0], [0, 1]],
        [[0, 1], [1, 0]],
        [[1, 0], [1, 0]],
        [[1, 1], [0, 1]],
    ]
    res = list()
    for i in range(20): res += pat
    random.shuffle(res)
    return res

import nn


class myunit(nn.unit):
    def __init__(self, *args):
        nn.unit.__init__(self, *args)

    def evaluator(self, res, tar):
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


def demo():
    u = myunit(2, 30, 2)
    u.name = "test"
    u.comment = "demo for XOR"
    u.set_activation_func([functions.relu, functions.tanh], weight = [1, 1])
    u.cost_func = functions.logloss
    for _ in u.train(pat_train(), u.evaluate, (pat_eval(), 0)): pass


if __name__ == '__main__':
    demo()
