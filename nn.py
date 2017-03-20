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
import scipy
import itertools
from scipy.ndimage.interpolation import shift

sns.set(style="darkgrid", palette="muted", color_codes=True)


class unit(object):
    def __init__(self, *args):
        self.n_layers = list(args)
        if len(args) < 2:
            raise ValueError("wrong number of inputs")
        self.n_layers[0] = self.n_layers[0] + 1  # fisrt layer for constant

        self.weights = list()
        self.weights_buff = list()
        self.weights_mask = list()
        for n, n_next in zip(self.n_layers, self.n_layers[1:]):
            mtrx = utils.gen_matrix(n_next, n)
            mask = utils.gen_mask(n_next, n)
            self.weights.append(mtrx)
            self.weights_buff.append(mtrx)
            self.weights_mask.append(mask)

        self.signals = list()
        for n in range(len(self.n_layers)):
            self.signals.append(np.array([0.] * n))

        # hyper parameters
        self.alpha = 0.01  # alpha: learning rate, placehoder
        self.beta = 0.9
        self.gamma = 0.9
        self.activation_temp = 0.01
        self.delta = 0.
        self.epsilon = 0.3
        self.zeta = 0.
        self.eta = 0.

        self.rs = list()
        for n in self.n_layers[1:]:
            self.rs.append(np.array([10.] * n))

        self.name = "no name"
        self.const = 1
        self.generation = 0
        self.score = 0
        self.sleep_count = 100

        # default settnig
        self.funcs = list()
        self.set_activation_func([functions.tanh, functions.relu])
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

    def aneal(self, how, *args):
        if how == "gaussian":
            mean, sig = args
            for cnt, w in enumerate(self.weights):
                self.weights[cnt] += self.weights_mask[cnt] * np.array([np.random.normal(mean, sig, w.shape[1]) for i in range(w.shape[0])])
        elif how == "random":
            vmin, vmax = args
            for cnt, w in enumerate(self.weights):
                self.weights[cnt] += self.weights_mask[cnt] * np.array([[utils.rand(-vmin, vmax) for i in range(w.shape[1])] for j in range(w.shape[0])])

    def sleep(self, how="type1"):
        if how == "type0":
            means = np.array([w.mean() for w in self.weights])
            sigs = np.array([w.std() for w in self.weights])
            self.weights = [np.logical_or(w < mean - sig * self.zeta, w > mean + sig * self.zeta) * w for w, mean, sig in zip(self.weights, means, sigs)]

        elif how == "type1":
            th = lambda x: np.sqrt(min([1, x]))
            th = np.vectorize(th)
            means = np.array([w.mean() for w in self.weights])
            sigs = np.array([w.std() for w in self.weights])
            self.weights = [th(abs(w  - mean / 0.5 / sig)) * w for w, mean, sig in zip(self.weights, means, sigs)]

    def forward_propagation(self, inputs):
        inputs = np.append(inputs, [self.const])
        if len(inputs) != self.n_layers[0]:
            raise ValueError("wrong number of inputs")
        # activate input node
        self.signals[0] = inputs
        for i in range(len(self.signals[1:])):
            self.signals[i + 1] = np.array([f(j) for f, j in zip(self.funcs[i][0], np.dot(self.weights[i], self.signals[i]))])

        return self.cost_func.func(self.signals[-1])

    def forward_propagation_(self, inputs, dropout=1):
        inputs = np.append(inputs, [self.const])
        if len(inputs) != self.n_layers[0]:
            raise ValueError("wrong number of inputs")

        # activate input node
        self.signals[0] = inputs
        for i in range(len(self.signals[1:])):
            self.signals[i + 1] = np.array([f(j) for f, j in zip(self.funcs[i][0], utils.dropout_output(self.weights[i], self.signals[i]))])

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
            self.weights[-n-1] += (self.alpha / (1 + self.epsilon * epoch + (1+n)**-1) * np.array([i * self.signals[-n-2] for i in delta / np.sqrt(self.rs[-n-1] + 1E-4)]) + self.gamma * (self.weights[-n-1] - self.weights_buff[-n-1]))
        self.weights_buff = buff

        error_in = np.dot(self.weights[0].T, delta)

        return self.cost_func.cost(self.signals[-1], targets, ), error_in

    def evaluate(self, patterns, save=0):
        cnt, corrct = 0, 0
        res = np.zeros(self.n_layers[-1] ** 2).reshape(self.n_layers[-1], self.n_layers[-1])
        for p in patterns:
            ans = self.forward_propagation(p[0])
            corrct += self.evaluator(ans, p[1])
            res[list(p[1]).index(max(p[1])), list(ans).index(max(ans))] += 1
            cnt += 1
        print("-"*100)
        print("%d / %d, raito%s" % (corrct, cnt, round(corrct / float(cnt) * 100, 2)))
        print(pd.DataFrame(res, columns=["pred%s" % i for i in range(self.n_layers[-1])], index=["corr%s" % i for i in range(self.n_layers[-1])]))
        print("-"*100)
        self.score = round(corrct / float(cnt) * 100, 2)
        if save:
            self.save()
        return corrct / float(cnt)

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
            self.weights_mask = [~utils.gen_mask(*w.shape) + self.delta for w in ob.weights_mask]
            self.funcs = ob.funcs
            self.rs = ob.rs
            self.generation = ob.generation + 1
            self.score = ob.score
            self.alpha, self.beta, self.gamma = ob.alpha, ob.beta, ob.gamma
            try:
                self.delta, self.epsilon, self.zeta, self.eta = ob.delta, ob.epsilon, ob.zeta, ob.eta
            except:
                pass
            print("copied % s" % f)

    def reset_mask(self):
        self.weights_mask = [utils.gen_mask(*w.shape) for w in self.weights_mask]

    def reproduce(self, unit, new_unit):
        for cnt, (w_dad, w_mom, w_dad_mask, w_mom_mask) in enumerate(zip(self.weights, unit.weights, self.weights_mask, unit.weights_mask)):
            new_unit.weights[cnt] = utils.merge_matrix(w_dad, w_mom, new_unit.weights[cnt].shape)
            new_unit.weights_mask[cnt] = utils.merge_matrix_mask(w_dad_mask, w_mom_mask, new_unit.weights_mask[cnt].shape)
        new_unit.alpha = (self.alpha + unit.alpha) / 2
        new_unit.generation = max([self.generation + unit.generation]) + 1
        for n in range(len(new_unit.funcs)):
            for i in range(new_unit.funcs[n].shape[1]):
                name = str()
                if n <= len(unit.funcs) - 1:
                    if i < unit.funcs[n].shape[1]: name = unit.funcs[n][-1, i]
                if i < self.funcs[n].shape[1] and not name: name = self.funcs[n][-1, i]
                new_unit.funcs[n][0, i], new_unit.funcs[n][1, i], new_unit.funcs[n][2, i] = utils.gen_func(random.choice([name]))
        new_unit.name = unit.name + "jr"
        return new_unit

    def check_progress(self, cnt, error, th=1e-7):
        x = np.array(list(range(len(error))))
        y = np.array(list(error))
        a, _, _, _, _ = scipy.stats.linregress(x, y)
        print("cnt, std, mean, slope -> %s, %s, %s, %s " % (cnt, np.array(list(error)).std(), np.array(list(error)).mean(), a), end="")
        if np.array(list(error)).std() < 0.0001:
            print("saturated")
            return StopIteration
        if np.array(list(error)).mean() > 10:
            print("overflow by mean")
            return StopIteration
        if list(error)[-1] != list(error)[-1]:
            print("overflow by nan")
            return StopIteration
        if a > th and self.sleep_count <= 0:
            print("ovverflow trend..go into sleep", end="")
            self.alpha *= 0.33
            self.alpha = max(1e-7, self.alpha)
            self.sleep_count = 100
        print("ok")

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

    def convolve(self, inputs, masksize=1):
        size = int(np.sqrt(inputs.shape[0]))
        arr = inputs.reshape(size, size)
        res = np.zeros(size*size).reshape(size, size)
        cnt = 0
        for a in itertools.product(list(range(-masksize, masksize+1)), list(range(-masksize, masksize+1))):
            res += shift(arr, a, cval=0)
            cnt +=1
        res /= cnt
        for i in range(masksize):
            res = np.delete(res, 0, axis=0)
            res = np.delete(res, -1, axis=0)
            res = np.delete(res, 0, axis=1)
            res = np.delete(res, -1, axis=1)
        return res.reshape((size - 2 * masksize)**2, 1)

    def describe(self):
        print("\tsize ->", "-".join(map(str, self.n_layers)))
        print("\talpha, beta, gamma, zeta, eta, temp -> %-.5f, %-.2f, %-.2f, %-.2f, %-.2f, %-.5f" % (self.alpha, self.beta, self.gamma, self.zeta, self.eta, self.activation_temp))
        print("\tgeneration ->", self.generation)

    def train(self, patterns, eval_fanc=0, arg=0, epoch=10000, how="online, Momentumsgd", interval=1000, save=0):
        if how == "online, Momentumsgd":
            times = np.array([])
            for i in range(epoch):
                error = list()

                # train
                errors = dict()
                s = time.time()
                random.shuffle(patterns)
                for cnt, p in enumerate(patterns):
                    self.sleep_count -=1
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error_this, _ = self.back_propagation(targets, epoch)
                    error.append(error_this)
                    errors.update({error_this: p})
                    if cnt % 1000 == 0 and cnt != 0:
                        if self.check_progress(cnt, error, 1e-3) == StopIteration:
                            raise StopIteration

                # fukusyu
                (_, patterns_fukusyu) = zip(*sorted(errors.items(), key=lambda x: x[0], reverse=1))
                errors_fukusyu = dict()
                for cnt, p in enumerate(patterns_fukusyu[:len(patterns) // 3]):
                    inputs = p[0]
                    targets = p[1]
                    self.forward_propagation(inputs)
                    error_this, _ = self.back_propagation(targets, epoch)
                    errors_fukusyu.update({error_this: p})
                    if cnt % 1000 == 0 and cnt != 0:
                        if self.check_progress(cnt, errors_fukusyu, 1e-4) == StopIteration:
                            raise StopIteration
                error = sum(error) / len(patterns)
                times = np.append(times, time.time() - s)
                if i % interval == 0:
                    print("epoch%s, error%-.5f, sec/epoc %-.3fsec, time remains %-.1fsec" % (i, error, time.time() - s, times.mean() * (epoch - i)))
                    if eval_fanc:
                        yield i, error, eval_fanc(*arg)
                    else:
                        yield i, error
                self.generation += 1

            if save:
                self.save()
            f = open("log.txt", "a")
            f.write("%s " % self.name + " ".join(map(str, self.n_layers)))
            f.write(" %s %s %s %s %s %s %s %s " % (self.alpha, self.beta, self.gamma, self.delta, self.epsilon, self.zeta, self.eta, self.activation_temp))
            f.write("%s %s %s" % (self.generation, len(patterns), self.score))
            f.write("\n")

        elif how == "batch":
            pass

if __name__ == '__main__':
    pass
