import nn
import utils
import numpy as np


def check_nn_gen_delault_connection():
    n = nn.unit(2, 2, 2)
    n.gen_default_connection(10, 0)
    print(n.connection)


def demo_singla_unit():
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
    n = nn.unit(3, 4, 1)
    n.set_pattern(pat)
    # n.initialization("random")
    n.animation()
    n.evaluate(pat)


def anonymous():
    a = np.array([[1, 0], [0, 2]])
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
    # check_nn_gen_delault_connection()
    # anonymous()
    demo_singla_unit()
    # check_ematrix()
