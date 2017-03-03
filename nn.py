# -*- coding: utf-8 -*-
import numpy as np
import random
import seaborn as sns
import utils
import functions
import pickle
import datetime
import os
import time
import pandas as pd

sns.set(style="darkgrid", palette="muted", color_codes=True)


class link(object):
    def __init__(self, n_input, n_output):
        self.layers = list()
        self.interposers = list()
        self.n_input = n_input
        self.n_output = n_output
        self.generation = 0
        self.score = 0
        self.design()

    def design(self):
        """
        Override and desribe your one links
        """
        pass

    def save(self):
        now = datetime.datetime.now()
        with open('./pickle/gen%s_score%s_%s_%02d%02d_%02d%02d%02d' % (self.generation, str(self.score).replace(".", "p"), now.year, now.month, now.day, now.hour, now.minute, now.second), mode='wb') as f:
            pickle.dump(self, f)
            print("saved as % s" % f.name)

    def clone(self, fp):
        with open(fp, mode='rb') as f:
            ob = pickle.load(f)
            self.layers = ob.layers
            self.interposers = ob.interposers
            self.n_input = ob.n_input
            self.n_output = ob.n_output
            self.generation = ob.generation
            self.score = ob.score

    def get_latest(self):
        res = None
        for f in os.listdir("./pickle"):
            if not res:
                res = f
                continue
            if os.path.getmtime("./pickle/%s" % res) < os.path.getmtime("./pickle/%s" % f):
                res = f
        print("copied % s" % res)
        self.clone("./pickle/%s" % res)

    def set_pattern(self, pat):
        self.pattern = pat

    def forward_propagation(self, inputs):
        self.interposers[0].set_signal(inputs)
        for layer, interposer, interposer_next in zip(self.layers, self.interposers, self.interposers[1:]):
            output_thislayer = np.array([])
            for u in layer:
                out = u.forward_propagation(u.convey_signal_foward_propagation(interposer.get_signal()))
                output_thislayer = np.append(output_thislayer, out)
            interposer_next.set_signal(output_thislayer)
        return self.interposers[-1].get_signal()

    def back_propagation(self, targets):
        error = 0
        for cnt, (layer, interposer, interposer_next) in enumerate(zip(self.layers[::-1], self.interposers[::-1], self.interposers[::-1][1:])):
            sindex = 0
            targets_next = np.array([])
            for u in layer:
                targets_thisu = targets[sindex: sindex + u.n_output]
                error_out, error_in = u.back_propagation(targets_thisu)
                if cnt == 0: error += error_out
                targets_next = np.append(targets_next, u.convey_signal_back_bropagation(error_in))
            targets = targets_next
        return error

    def train(self, patterns, epoch=10000, how="online", save=0, interval=1000):
        if how == "online":
            times = np.array([])
            for i in range(epoch):
                error = 0.
                s = time.time()
                for p in patterns:
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error_this = self.back_propagation(targets)
                    error = error + error_this
                times = np.append(times, time.time() - s)
                if i % interval == 0:
                    print("epoch%s, error%-.5f, sec/epoc %-.3fsec, time remains %-.1fsec" % (i, error, time.time() - s, times.mean() * (epoch - i)))
                    yield i, error
                if i % 10 == 0 and save:
                    self.save()
            self.score = round(error, 5)
        elif how == "batch":
            pass
        if not os.path.exists("./pickle"):
            os.mkdir("./pickle")

    def evaluate(self, patterns):
        cnt, corrct = 0, 0
        for p in patterns:
            ans = self.forward_propagation(p[0])
            corrct += self.evaluator(ans, p[1])
            cnt += 1
        print("%d / %d, raito%s" % (corrct, cnt, round(corrct / float(cnt) * 100, 2)))
        # self.describe()

    def evaluator(self, res, tar):
        return 0

    def describe(self):
        print("gen, score -> %s, %s" % (self.generation, self.score))
        for u in utils.flatten(self.layers):
            print("name -> %s" % u.name)
            print("\tsize %s-%s-%s" % (u.n_input, u.n_interm, u.n_output))
            print("\talpha -> %s" % u.alpha)
            print("\tfuncs input-interm -> %s" % ",".join(u.func_names_interm))
            print("\tfuncs input-interm -> %s" % ",".join(u.func_names_output))


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

        self.alpha = abs(np.random.normal(0., 0.01))  # alpha: learning rate
        self.r = 0.01

        self.connection, self.connection_inv = np.array([]), np.array([])

        self.buffer_unit = buff
        self.fixed = fixed
        self.name = "no name"
        self.const = 1

        # default settnig
        self.set_activation_func([functions.tanh, functions.relu])
        if not self.buffer_unit:
            self.initialization("gaussian", 0, 2)
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
        self.connection_inv = np.linalg.inv(np.dot(self.connection.T, self.connection))

    def convey_signal_foward_propagation(self, signal):
        return np.dot(self.connection, signal)

    def convey_signal_back_bropagation(self, error):
        return np.dot(self.connection_inv, np.dot(self.connection.T, error[:-1]))

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
        inputs = np.append(inputs, [self.const])
        if len(inputs) != self.n_input:
            raise ValueError("wrong number of inputs")

        # activate input node
        self.signal_input = inputs

        # hidden activations
        self.signal_interm = np.array([f(i) for f, i in zip(self.func_interm, np.dot(self.weights_input_to_interm, self.signal_input))])

        # output activations
        self.signal_output = np.array([f(i) for f, i in zip(self.func_output, np.dot(self.weights_interm_to_output, self.signal_interm))])

        return self.signal_output

    def back_propagation(self, targets):
        r = 0.
        if len(targets) != self.n_output:
            raise ValueError("wrong number of target values")

        # update interm - output weights
        error_output = self.cost_func.derv(self.signal_output, targets)
        r += (self.cost_func.derv(self.signal_output, targets)).sum() ** 2
        delta_output = np.array([f(i) * j for f, i, j in zip(self.dfunc_output, self.signal_output, error_output)])

        # update input/interm weights
        error_interm = np.dot(self.weights_interm_to_output.T, delta_output)
        delta_interm = np.array([f(i) * j for f, i, j in zip(self.dfunc_interm, self.signal_interm, error_interm)])
        self.weights_input_to_interm += self.alpha / np.sqrt(r + 1E-4) * np.array([i * self.signal_input for i in delta_interm])

        error_input = np.dot(self.weights_input_to_interm.T, delta_interm)

        return self.cost_func.func(targets, self.signal_output), error_input

    def evaluate(self, patterns):
        cnt, corrct = 0, 0
        for p in patterns:
            ans = self.forward_propagation(p[0])
            print(p[0], '->', ans, p[1])
            cnt += 1
            if np.linalg.norm(ans - p[1]) < 0.1:
                corrct += 1
        print("reslut -> %d / %d, raito%f" % (corrct, cnt, round(corrct / float(cnt), 3)))
        self.describe()

    def describe(self):
        print("size %s-%s-%s" % (self.n_input, self.n_interm, self.n_output))
        print("alpha ->", self.alpha)
        print("funcs input-interm ->", ",".join(self.func_names_interm))
        print("funcs input-interm ->", ",".join(self.func_names_output))

    def train(self, patterns, epoch=10000, how="online"):
        if how == "online":
            times = np.array([])
            for i in range(epoch):
                error = 0.0
                s = time.time()
                for p in patterns:
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error_this, _ = self.back_propagation(targets)
                    error = error + error_this
                times = np.append(times, time.time() - s)
                if i % 1 == 0:
                    print("epoch%s, error%-.5f, sec/epoc %-.3fsec, time remains %-.1fsec" % (i, error, time.time() - s, times.mean() * (epoch - i)))
                    yield i, error

        elif how == "batch":
            pass

if __name__ == '__main__':
    pass
