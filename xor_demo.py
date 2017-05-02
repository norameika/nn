import random
import nn


def pat_train():
    pat = [
        [[[0, 0], ], [0, 1]],
        [[[0, 1], ], [1, 0]],
        [[[1, 0], ], [1, 0]],
        [[[1, 1], ], [0, 1]],
    ]
    res = list()
    for i in range(1000): res += pat
    random.shuffle(res)
    return res


def pat_eval():
    pat = [
        [[[0, 0], ], [0, 1]],
        [[[0, 1], ], [1, 0]],
        [[[1, 0], ], [1, 0]],
        [[[1, 1], ], [0, 1]],
    ]
    res = list()
    for i in range(20): res += pat
    random.shuffle(res)
    return res

def demo():
    mymodel = nn.model()
    mymodel.name = "test"
    mymodel.add_prop([nn.fc(2, 100)])
    mymodel.add_prop([nn.node(100)])
    mymodel.add_prop([nn.fc(100, 2)])
    mymodel.add_prop([nn.node_out(2)])

    mymodel.train(pat_train(), pat_eval(), 10)

if __name__ == '__main__':
    demo()