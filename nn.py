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
import copy


sns.set(style="darkgrid", palette="muted", color_codes=True)


class unit(object):
    def __init__(self, *args):
        self.n_layers = list(args)
        if len(args) < 2:
            raise ValueError("wrong number of inputs")
        self.n_layers[0] = self.n_layers[0] + 1  # fisrt layer for constant

        self.weights = list()
        self.weights_buff = list()
        for n, n_next in zip(self.n_layers, self.n_layers[1:]):
            mtrx = utils.gen_matrix(n_next, n)
            self.weights.append(mtrx)
            self.weights_buff.append(mtrx)

        self.signals = list()
        for n in range(len(self.n_layers)):
            self.signals.append(np.array([0.] * n))

        # hyper parameters
        self.alpha = abs(np.random.normal(0., 0.001))  # alpha: learning rate
        self.beta = 0.9
        self.gamma = 0.9

        self.rs = list()
        for n in self.n_layers[1:]:
            self.rs.append(np.array([10.] * n))

        self.name = "no name"
        self.const = 1
        self.generation = 0
        self.score = 0
        # default settnig
        self.funcs = list()
        self.set_activation_func([functions.tanh, functions.relu])
        self.initialization("gaussian", 0, 0.001)
        self.cost_func = functions.square_error

    def set_activation_func(self, funcs):
        for n in self.n_layers[1:]:
            """activation func, derivative of func, name"""
            self.funcs.append(np.array([[f.func, f.derv, f.name] for f in [random.choice(funcs) for i in range(n)]]).T)

    def initialization(self, how, *args):
        if how == "gaussian":
            mean, sig = args
            for cnt, w in enumerate(self.weights):
                self.weights[cnt] = np.array([np.random.normal(mean, sig, w.shape[1]) for i in range(w.shape[0])])
        elif how == "random":
            vmin, vmax = args
            for cnt, w in enumerate(self.weights):
                self.weights[cnt] = np.array([[utils.rand(-vmin, vmax) for i in range(w.shape[1])] for j in range(w.shape[0])])

    def forward_propagation(self, inputs):
        inputs = np.append(inputs, [self.const])
        if len(inputs) != self.n_layers[0]:
            raise ValueError("wrong number of inputs")

        # activate input node
        self.signals[0] = inputs
        for i in range(len(self.signals[1:])):
            self.signals[i + 1] = np.array([f(j) for f, j in zip(self.funcs[i][0], np.dot(self.weights[i], self.signals[i]))])

        return self.cost_func.func(self.signals[-1])

    def back_propagation(self, targets, epoch):
        """momentan SDG"""
        if len(targets) != self.n_layers[-1]:
            raise ValueError("wrong number of target values")
        error = self.cost_func.derv(self.signals[-1], targets)
        delta = 0
        buff = self.weights
        for n in range(len(self.signals[1:])):
            if n != 0: error = np.dot(self.weights[-n].T, delta)
            delta = np.array([f(i) for f, i in zip(self.funcs[-n-1][1], self.signals[-n-1])]) * error
            self.rs[-n-1] = self.beta * self.rs[-n-1] + (1 - self.beta) * (delta * delta).mean()

            self.weights[-n-1] += self.alpha * np.array([i * self.signals[-n-2] for i in delta / np.sqrt(self.rs[-n-1] + 1E-4)]) + self.gamma * (self.weights[-n-1] - self.weights_buff[-n-1])
        self.weights_buff = buff

        error_in = np.dot(self.weights[0].T, delta)

        return self.cost_func.cost(self.signals[-1], targets, ), error_in

    def evaluate(self, patterns, save=0):
        cnt, corrct = 0, 0
        for p in patterns:
            ans = self.forward_propagation(p[0])
            corrct += self.evaluator(ans, p[1])
            cnt += 1
        print("%d / %d, raito%s" % (corrct, cnt, round(corrct / float(cnt) * 100, 2)))
        self.score = round(corrct / float(cnt) * 100, 2)
        self.generation += 1
        if save:
            self.save()
        return corrct / float(cnt)
        # self.describe()

    def evaluator(self, res, tar):
        return 0

    def save(self):
        now = datetime.datetime.now()
        if not os.path.exists("./pickle"):
            os.mkdir("./pickle")
        with open('./pickle/%s_gen%s_score%s_%s_%02d%02d_%02d%02d%02d' % (self.name, self.generation, str(self.score).replace(".", "p"), now.year, now.month, now.day, now.hour, now.minute, now.second), mode='wb') as f:
            pickle.dump(self, f)
            print("saved as % s" % f.name)

    def clone(self, fp):
        with open(fp, mode='rb') as f:
            ob = pickle.load(f)
            self.n_layers = ob.n_layers
            self.name = ob.name
            self.weights = ob.weights
            self.weights_buff = ob.weights_buff
            self.funcs = ob.funcs
            self.rs = ob.rs
            self.generation = ob.generation + 1
            self.score = ob.score
            self.alpha, self.beta, self.gamma = ob.alpha, ob.beta, ob.gamma
            print("copied % s" % f)

    def reproduce(self, unit, new_unit):
        for cnt, (w_dad, w_mom) in enumerate(zip(self.weights, unit.weights)):
            new_unit.weights[cnt] = utils.merge_matrix(w_dad, w_mom, new_unit.weights[cnt].shape)
        new_unit.alpha = (self.alpha + unit.alpha) / 2
        new_unit.generation = max([self.generation + unit.generation]) + 1
        for n in range(len(new_unit.funcs)):
            for i in range(new_unit.funcs[n].shape[1]):
                names = list()
                if n <= len(unit.funcs) - 1:
                    if i < unit.funcs[n].shape[1]: names.append(unit.funcs[n][-1, i])
                    continue
                if i < self.funcs[n].shape[1]: names.append(self.funcs[n][-1, i])
                new_unit.funcs[n][0, i], new_unit.funcs[n][1, i], new_unit.funcs[n][2, i] = utils.gen_func(random.choice(names))
        new_unit.name = unit.name + "jr"
        return new_unit

    def get_latest(self):
        res = None
        for f in os.listdir("./pickle"):
            if not f.startswith(self.name): continue
            if not res:
                res = f
                continue
            if os.path.getmtime("./pickle/%s" % res) < os.path.getmtime("./pickle/%s" % f):
                res = f
        print("copied % s" % res)
        self.clone("./pickle/%s" % res)

    def describe(self):
        print("\tsize ->", "-".join(map(str, self.n_layers)))
        print("\talpha, beta, gamma -> %-.5f, %-.2f, %-.2f" % (self.alpha, self.beta, self.gamma))
        print("\tgeneration ->", self.generation)

    def train(self, patterns, eval_fanc=0, arg=0, epoch=10000, how="online, Momentumsgd", interval=1000, save=0):
        if how == "online, Momentumsgd":
            times = np.array([])
            for i in range(epoch):
                error = 0.0
                s = time.time()
                random.shuffle(patterns[::(-1) ** epoch])
                for p in patterns:
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error_this, _ = self.back_propagation(targets, epoch)
                    error = error + error_this
                error /= len(patterns)
                times = np.append(times, time.time() - s)
                if i % interval == 0:
                    print("epoch%s, error%-.5f, sec/epoc %-.3fsec, time remains %-.1fsec" % (i, error, time.time() - s, times.mean() * (epoch - i)))
                    if eval_fanc:
                        yield i, error, eval_fanc(*arg)
                    else:
                        yield i, error
            if save:
                self.save()

        elif how == "batch":
            pass

if __name__ == '__main__':
    pass
