# -*- coding: utf-8 -*-
import numpy as np
import random
from matplotlib import pylab as plt
import matplotlib.animation as animation
import seaborn as sns
import utils
import functions

sns.set(style="darkgrid", palette="muted", color_codes=True)


class link(object):
    def __init__(self, n_input, n_output):
        self.interposer_input = interposer(n_input)
        self.interposer_interm = interposer(4)
        self.interposer_output = interposer(1)

        self.layers= list()

        # design 1st layer
        unit00 = unit(2, 2, 2)
        unit00.gen_default_connection(n_input, 0)
        unit01 = unit(2, 2, 2)
        unit01.gen_default_connection(n_input, 2)
        self.layer.append([unit00, unit01])

        # design 2nd layer
        unit10 = unit(4, 4, 1)
        self.layers.append([unit10, ])

        self.interposers = list(self.interposer_input, self.interposer_interm, self.interposer_output)  # length of interposers should be len(self.layers) + 1

    def set_pattern(self, pat):
        self.pattern = pat

    def forward_propagation(self, inputs):
        self.interposers[0].set_signal(inputs)
        for layer, interposer, interposer_next in zip(self.layers, self.interposers, self.interposers[1:]):
            output_thislayer = np.array([])
            for unit in layer:
                out = unit.forward_propagation(unit.convey_signal_foward_propagation(self.interposers.get_signal()))
                np.append(output_thislayer, out)
            interposer_next.set_signal(output_thislayer)
        return self.interposers[-1].get_signal()

    def back_propagation(self, targets):
        for layer, interposer, interposer_next in zip(self.layers, self.interposers[::-1], self.interposers[::-1][1:]):
            sindex = 0
            targets_next = np.array([])
            for unit in layer:
                targets_thisunit = targets[sindex: sindex + unit.n_output]
                error_out, error_in = unit.back_propagation(targets_thisunit)
                np.append(targets_next, unit.convey_signal_back_bropagation(error_in))
            targets = targets_next
        return 1


class interposer(object):
    def __init__(self, n_pins):
        self.n_pins = n_pins
        self.singals = np.array([0] * n_pins)

    def set_signal(self, inputs):
        self.signals = inputs

    def get_signal(self):
        return self.signals


class unit(object):
    def __init__(self, n_input, n_interm, n_output, buff=0, fixed=0):
        self.n_input = n_input + 1  # 1 for constant
        self.n_interm = n_interm
        self.n_output = n_output

        self.weights_input_to_interm = utils.gen_matrix(self.n_interm, self.n_input)
        self.weights_interm_to_output = utils.gen_matrix(self.n_output, self.n_interm)

        self.signal_input = np.array([0.] * self.n_input)
        self.signal_interm = np.array([0.] * self.n_interm)
        self.signal_output = np.array([0.] * self.n_output)

        self.alpha = abs(np.random.normal(0, 0.01))  # alpha: learning rate

        self.connection, self.connection_inv = np.array([]), np.array([])

        self.buffer_unit = buff
        self.fixed = fixed

        # for animation
        self.fig, (self.ax0) = plt.subplots(1, 1)
        self.ani = 0
        self.line0, = self.ax0.plot(list(), list(), lw=0.5)
        self.line = [self.line0]
        self.xdata, self.ydata = list(), list()

        # default settnig
        self.set_activation_func([functions.tanh, functions.relu])
        if not self.buffer_unit:
            self.initialization("gaussian", 0, 2.)
        else:
            self.initialization("buff")

        self.cost_func = functions.square_error

    def set_activation_func(self, funcs):
        self.func_interm, self.dfunc_interm, self.func_names_interm = zip(*[[f.func, f.derv, f.name] for f in [random.choice(funcs) for i in range(self.n_interm)]])
        self.func_output, self.dfunc_output, self.func_names_output = zip(*[[f.func, f.derv, f.name] for f in [random.choice(funcs) for i in range(self.n_output)]])

    def set_pattern(self, pat):
        self.pattern = pat

    def set_cost_func(self, func):
        self.cost_func = func

    def gen_default_connection(self, n_input, sindex):
        if n_input + sindex < self.n_input - 1:
            raise ValueError("wrong number of inputs")
        self.connection = utils.gen_matrix(self.n_input - 1, n_input, fill=0.)
        for i, j in enumerate(range(sindex, sindex + self.n_input - 1)):
            self.connection[i, j] = 1.
        self.connection_inv = np.linalg.inv(unit.connection, unit.connection.T)

    def convey_signal_foward_propagation(self, signal):
        return np.dot(unit.connection, signal)

    def convey_signal_back_bropagation(self, error):
        return np.dot(np.dot(error, unit.connection.T), self.connection_inv)

    def initialization(self, how, *args):
        if how == "gaussian":
            mean, sig = args
            self.weights_input_to_interm = np.array([np.random.normal(mean, sig, self.n_input) for i in range(self.n_interm)])
            self.weights_interm_to_output = np.array([np.random.normal(mean, sig, self.n_interm) for i in range(self.n_output)])
        elif how == "random":
            self.weights_input_to_interm = np.array([[utils.rand(-0.1, 0.2) for i in range(self.n_input)] for j in range(self.n_interm)])
            self.weights_interm_to_output = np.array([[utils.rand(-2.0, 2.0) for i in range(self.n_interm)] for j in range(self.n_output)])
        elif how == "buff":
            if self.n_input - 1 != self.n_interm or self.n_interm != self.n_output:
                raise ValueError("wrong number of inputs, interm, ouput")
            self.weights_input_to_interm = utils.gen_ematrix(self.n_input, 1)
            self.weights_interm_to_output = utils.gen_ematrix(self.n_output, 0)

    def forward_propagation(self, inputs):
        inputs = np.append(inputs, [1])
        if len(inputs) != self.n_input:
            raise ValueError("wrong number of inputs")

        # input activations
        self.signal_input = inputs

        # hidden activations
        self.signal_interm = np.array([f(i) for f, i in zip(self.func_interm, np.dot(self.weights_input_to_interm, self.signal_input))])

        # output activations
        self.signal_output = np.array([f(i) for f, i in zip(self.func_output, np.dot(self.weights_interm_to_output, self.signal_interm))])

        return self.signal_output

    def back_propagation(self, targets):
        alpha = self.alpha
        if len(targets) != self.n_output:
            raise ValueError("wrong number of target values")

        # update interm - output weights
        error_output = self.cost_func.derv(self.signal_output, targets)
        delta_output = np.array([f(i) * j for f, i, j in zip(self.dfunc_output, self.signal_output, error_output)])
        self.weights_interm_to_output += alpha * np.array([i * self.signal_interm for i in delta_output])

        # update input - interm weights
        error_interm = np.dot(self.weights_interm_to_output.T, delta_output)
        delta_interm = np.array([f(i) * j for f, i, j in zip(self.dfunc_interm, self.signal_interm, error_interm)])
        self.weights_input_to_interm += alpha * np.array([i * self.signal_input for i in delta_interm])

        error_input = np.dot(self.weights_input_to_interm.T, delta_interm)

        return self.cost_func.func(targets, self.signal_output), error_input

    def evaluate(self, patterns):
        for p in patterns:
            print(p[0], '->', self.forward_propagation(p[0]), p[1])
        self.describe()

    def describe(self):
        print("size %s-%s-%s" % (self.n_input, self.n_interm, self.n_output))
        print("alpha ->", self.alpha)
        print("funcs input-interm ->", ",".join(self.func_names_interm))
        print("funcs input-interm ->", ",".join(self.func_names_output))

    def train(self, patterns, epoch=10000, how="online"):
        if how == "online":
            for i in range(epoch):
                error = 0.0
                for p in patterns:
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error_this, _ = self.back_propagation(targets)
                    error = error + error_this
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


if __name__ == '__main__':
    pass
