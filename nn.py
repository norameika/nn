# -*- coding: utf-8 -*-
import numpy as np
import random
from matplotlib import pylab as plt
import matplotlib.animation as animation
import seaborn as sns
import fanctions
import utils

sns.set(style="darkgrid", palette="muted", color_codes=True)

# class link(object):
#     def __init__(self):


class interposer(object):
    def __init__(self, n_pins):
        self.n_pins = n_pins
        self.singals = np.array([0] * n_pins)

    def set_signal(self, input, n_start):
        pass


class nn(object):
    def __init__(self, n_input, n_interm, n_output):
        self.n_input = n_input + 1
        self.n_interm = n_interm
        self.n_output = n_output

        self.weights_input_to_interm = utils.gen_matrix(self.n_interm, self.n_input)
        self.weights_interm_to_output = utils.gen_matrix(self.n_output, self.n_interm)

        self.signal_input = np.array([0.] * self.n_input)
        self.signal_interm = np.array([0.] * self.n_interm)
        self.signal_output = np.array([0.] * self.n_output)

        self.alpha = abs(np.random.normal(0., 0.01))  # alpha: learning rate

        self.fig, (self.ax0) = plt.subplots(1, 1)
        self.ani = 0

        # intialize two line objects (one in each axes)
        self.line0, = self.ax0.plot([], [], lw=0.5)
        self.line = [self.line0]
        self.xdata, self.ydata = [], []

    def set_activation_fanc(self, fancs):
        self.fanc_interm, self.dfanc_interm, self.fanc_names_interm = zip(*[[f.fanc, f.derv, f.name] for f in [random.choice(fancs) for i in range(self.n_interm)]])
        self.fanc_output, self.dfanc_output, self.fanc_names_output = zip(*[[f.fanc, f.derv, f.name] for f in [random.choice(fancs) for i in range(self.n_output)]])

    def set_pattern(self, pat):
        self.pattern = pat

    def initialization(self, how, *args):
        if how == "gaussian":
            mean, sig = args
            self.weights_input_to_interm += np.array([np.random.normal(mean, sig, self.n_input) for i in range(self.n_interm)])
            self.weights_interm_to_output += np.array([np.random.normal(mean, sig, self.n_interm) for i in range(self.n_output)])
        elif how == "random":
            self.weights_input_to_interm += np.array([[utils.rand(-0.1, 0.2) for i in range(self.n_input)] for j in range(self.n_interm)])
            self.weights_interm_to_output += np.array([[utils.rand(-2.0, 2.0) for i in range(self.n_interm)] for j in range(self.n_output)])

    def forward_propagation(self, inputs):
        inputs = np.append(inputs, [1])
        if len(inputs) != self.n_input:
            raise ValueError('wrong number of inputs')

        # activate input node
        self.signal_input = inputs

        # activate interm node
        self.signal_interm = np.array([f(i) for f, i in zip(self.fanc_interm, np.dot(self.weights_input_to_interm, self.signal_input))])

        # activate output node
        self.signal_output = np.array([f(i) for f, i in zip(self.fanc_output, np.dot(self.weights_interm_to_output, self.signal_interm))])

        return self.signal_output

    def back_propagation(self, targets):
        alpha = self.alpha
        if len(targets) != self.n_output:
            raise ValueError('wrong number of target values')

        # update interm/output weights
        error_output = targets - self.signal_output
        delta_output = np.array([f(i) * j for f, i, j in zip(self.dfanc_output, self.signal_output, error_output)])
        self.weights_interm_to_output += alpha * np.array([i * self.signal_interm for i in delta_output])

        # update input/interm weights
        error_interm = np.dot(self.weights_interm_to_output.T, delta_output)
        delta_interm = np.array([f(i) * j for f, i, j in zip(self.dfanc_interm, self.signal_interm, error_interm)])
        self.weights_input_to_interm += alpha * np.array([i*self.signal_input for i in delta_interm])
        return sum([i ** 2 for i in error_output])

    def evaluate(self, patterns):
        for p in patterns:
            print(p[0], '->', self.forward_propagation(p[0]), p[1])
        self.describe()

    def describe(self):
        print("size %s-%s-%s" % (self.n_input, self.n_interm, self.n_output))
        print("alpha ->", self.alpha)
        print("fancs input/interm ->", ",".join(self.fanc_names_interm))
        print("fancs interm/ouput ->", ",".join(self.fanc_names_output))

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
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]],
    ]

    # create a network with two input, two hidden, and one output nodes
    n = nn(2, 3, 1)
    n.set_activation_fanc([fanctions.tanh, fanctions.relu])
    n.set_pattern(pat)
    n.initialization("gaussian", 0, 2.)
    # n.initialization("random")
    n.animation()
    # n.initialization("random")
    # train it with some patterns
    # test it
    # for d in n.train(n.pattern, epoch=10000):
    #     i, error = d
    #     print(i, 'error %-.5f' % error)
    n.evaluate(pat)

if __name__ == '__main__':
    demo()
