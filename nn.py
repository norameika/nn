# -*- coding: utf-8 -*-
import numpy as np
import random
from matplotlib import pylab as plt
import matplotlib.animation as animation
import seaborn as sns

sns.set(style="darkgrid", palette="muted", color_codes=True)

def _rand(a, b):
    return (b-a)*random.random()+a


def _gen_matrix(i, j, fill=0.):
    matrix = []
    for i in range(i):
        matrix.append([fill] * j)
    return np.array(matrix)


def _tanh(x):
    return np.tanh(x)


def _dtanh(y):
    """d (tanh(x)) / dx = 1-tanh2(x)
    """
    return 1. - y**2


def _relu(x):
    return min(1, max(0, x))


def _drelu(y):
    return int(y >= 0)


class nn(object):
    def __init__(self, n_input, n_interm, n_output):
        self.n_input = n_input + 1
        self.n_interm = n_interm
        self.n_output = n_output

        self.weights_input_to_interm = _gen_matrix(self.n_interm, self.n_input)
        self.weights_interm_to_output = _gen_matrix(self.n_output, self.n_interm)

        self.signal_input = np.array([0.] * self.n_input)
        self.signal_interm = np.array([0.] * self.n_interm)
        self.signal_output = np.array([0.] * self.n_output)

        self.alpha = 0.01  # alpha: learning rate

        self.fig, (self.ax0) = plt.subplots(1, 1)
        self.ani = 0

        # intialize two line objects (one in each axes)
        self.line0, = self.ax0.plot([], [], lw=0.5)
        self.line = [self.line0]
        self.xdata, self.ydata = [], []

    def set_activation_fanc(self, fanc, dfanc):
        self.fanc = fanc
        self.dfanc = dfanc

    def set_pattern(self, pat):
        self.pattern = pat

    def initialization(self, how, *args):
        if how == "gaussian":
            mean, sig = args
            self.weights_input_to_interm += np.array([np.random.normal(mean, sig, self.n_input) for i in range(self.n_interm)])
            self.weights_interm_to_output += np.array([np.random.normal(mean, sig, self.n_interm) for i in range(self.n_output)])
        elif how == "random":
            self.weights_input_to_interm += np.array([[_rand(-0.1, 0.2) for i in range(self.n_input)] for j in range(self.n_interm)])
            self.weights_interm_to_output += np.array([[_rand(-2.0, 2.0) for i in range(self.n_interm)] for j in range(self.n_output)])

    def forward_propagation(self, inputs):
        fanc = self.fanc
        inputs = np.append(inputs, [0])
        if len(inputs) != self.n_input:
            raise ValueError('wrong number of inputs')

        # input activations
        self.signal_input = inputs

        # hidden activations
        self.signal_interm = np.array([fanc(i) for i in np.dot(self.weights_input_to_interm, self.signal_input)])

        # output activations
        self.signal_output = np.array([fanc(i) for i in np.dot(self.weights_interm_to_output, self.signal_interm)])

        return self.signal_output

    def back_propagation(self, targets):
        alpha = self.alpha
        fanc = self.dfanc
        if len(targets) != self.n_output:
            raise ValueError('wrong number of target values')

        # update interm - output weights
        error_output = targets - self.signal_output
        delta_output = np.array([fanc(i) * j for i, j in zip(self.signal_output, error_output)])
        self.weights_interm_to_output += alpha * np.array([i * self.signal_interm for i in delta_output])

        # update input - interm weights
        error_interm = np.dot(self.weights_interm_to_output.T, delta_output)
        delta_interm = np.array([fanc(i) * j for i, j in zip(self.signal_interm, error_interm)])
        self.weights_input_to_interm += alpha * np.array([i*self.signal_input for i in delta_interm])
        return sum([i ** 2 for i in error_output])

    def evaluate(self, patterns):
        for p in patterns:
            print(p[0], '->', self.forward_propagation(p[0]), p[1])

    def train(self, patterns, epoch=1000, how="online"):
        if how == "online":
            for i in range(epoch):
                error = 0.0
                for p in patterns:
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error = error + self.back_propagation(targets)
                if i % 1000 == 0:
                    yield i, error

        elif how == "batch":
            pass

    def run(self, data):
        # update the data
        epoch, error = data
        # self.xdata, self.y0data, = [], []
        self.xdata.append(epoch)
        self.ydata.append(error)

        self.ax0.set_ylim(-max(self.ydata) * 0.1, max(self.ydata) * 1.2)
        self.ax0.set_xlim(-max(self.xdata) * 0.1, max(self.xdata) * 1.2)

        self.line[0].set_data(self.xdata, self.ydata)
        self.ax0.set_title("%-.5f" % error)
        self.ax0.set_xlabel("epochs")
        for ax in [self.ax0]:
            ax.figure.canvas.draw()
        return self.line

    def animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.run, self.arrange_for_animation, interval=100, blit=False, repeat=False)
        plt.show()
        return self.ani

    def arrange_for_animation(self):
        for d in self.train(self.pattern, epoch=100000):
            i, error = d
            print(i, 'error %-.5f' % error)
            yield d

    def show(self, **kwargs):
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot([[], []], lw=1)
        plt.title("test", fontsize=16)
        plt.show()


def demo():
    # Teach network XOR function
    pat = [
        [[0, 0, 0], [0]],
        [[0, 0, 1], [1]],
        [[0, 1, 0], [1]],
        [[0, 1, 1], [0]],
        [[1, 0, 0], [1]],
        [[1, 0, 1], [0]],
        [[1, 1, 0], [0]],
        [[1, 1, 1], [1]],
    ]

    # create a network with two input, two hidden, and one output nodes
    n = nn(3, 3, 1)
    n.set_activation_fanc(_tanh, _dtanh)
    n.set_pattern(pat)
    n.initialization("gaussian", 0, 1.)
    n.alpha = 0.01
    # n.initialization("random")
    n.animation()
    # n.initialization("random")pipi
    # train it with some patterns
    # test it
    # for d in n.train(n.pattern, epoch=10000):
    #     i, error = d
    #     print(i, 'error %-.5f' % error)
    n.evaluate(pat)

if __name__ == '__main__':
    demo()
