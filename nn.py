# -*- coding: utf-8 -*-
import numpy
import random
import seaborn
try:
    from . import utils
    from . import functions
except:
    import utils
    import functions
import pickle
import datetime
import os
import time
import pandas
import scipy
import copy

seaborn.set(style="darkgrid", palette="muted", color_codes=True)


class node(object):
    def __init__(self, n, dropout, end):
        self.n = n
        self.signal = numpy.array([0] * n)
        self.dropout = dropout
        self.mask = numpy.array([1] * n)

    def forward_propagation(self, funcs, weight, evaluate):
        if self.dropout and not evaluate: self.update_dropout_mask(self.n)
        else: self.mask = numpy.array([1] * self.n)
        return numpy.array([f(j) for f, j in zip(funcs, numpy.dot(weight, self.mask * self.signal))])

    def back_propagation(self, error, funcs_upper_step, rs, beta, alpha, epsilon, signal_upper_step, gamma, weight, weight_buff, epoch, step, node_upper, evaluate=0):
        if self.dropout and not evaluate: self.update_dropout_mask(self.n)
        delta = numpy.array([f(i) for f, i in zip(funcs_upper_step, node_upper.signal)]) * error
        delta_mask = delta * node_upper.mask
        rs = beta * rs + (1 - beta) * (delta_mask * delta_mask).mean()
        weight_delta = (alpha / (1 + epsilon * (epoch + (1+step*2)**-2)) * numpy.array([i * self.signal for i in delta_mask / numpy.sqrt(rs + 1E-4)]) + gamma * (weight - weight_buff))
        return weight_delta, rs, delta

    def set_signal(self, signal):
        self.signal = signal

    def update_dropout_mask(self, n):
        mask = numpy.array([i < 1 and i > -1 for i in numpy.random.normal(0, 1, n)]).astype(numpy.int16)
        self.mask = mask / 0.6825


class unit(object):
    def __init__(self, *args, **kwargs):
        self.n_layers = list(args)
        if len(args) < 2:
            raise ValueError("wrong number of inputs")
        self.n_layers[0] = self.n_layers[0] + 1  # fisrt layer for constant

        self.weights = [utils.gen_matrix(n_next, n) for n, n_next in zip(self.n_layers, self.n_layers[1:])]
        self.weights_buff = copy.deepcopy(self.weights)

        if "dropout" in kwargs.keys(): self.dropout = kwargs["dropout"]
        else: self.dropout = 0
        self.nodes = [node(n, dropout=self.dropout, end=int(cnt == 0)) for cnt, n in enumerate(self.n_layers)]

        # hyper parameters
        self.alpha = 10 ** (-3 + numpy.random.normal(0, 1))
        self.beta = min(1, numpy.random.normal(0.8, 0.05))
        self.gamma = min(1, numpy.random.normal(0.8, 0.05))
        self.delta = 10 ** (-4 + numpy.random.normal(0, 1))
        self.epsilon =  abs(numpy.random.normal(0.3, 0.1))
        self.rs = [numpy.array([10.] * n) for n in self.n_layers[1:]]

        self.comment = "no comment"
        self.name = "no name"
        self.const = 1
        self.generation = 0
        self.score = 999
        self.score_prvs = 999

        # default settnig
        self.funcs = list()
        self.set_activation_func([functions.tanh, functions.relu])
        self.cost_func = functions.square_error
        self.initialization("gaussian", 0, self.delta)

        if "pre_units" in kwargs.keys():
            self.pre_units = kwargs["pre_units"]
            self.input_index = kwargs["input_index"]
            self.unit_type = "connected"
        else:
            self.unit_type = "isolated"

    def set_activation_func(self, funcs, **kwargs):
        if "weight" in kwargs.keys(): weight = kwargs["weight"]
        else: weight = [1] * len(funcs)
        for n in self.n_layers[1:]:
            self.funcs.append(numpy.array([[f.func, f.derv, f.name] for f in [random.choice(funcs) for i in range(n)]]).T)

    def initialization(self, how, *args):
        if how == "gaussian":
            mean, sig = args
            for cnt, w in enumerate(self.weights):
                self.weights[cnt] = numpy.array([numpy.random.normal(mean, sig, w.shape[1]) for i in range(w.shape[0])])
        elif how == "random":
            vmin, vmax = args
            for cnt, w in enumerate(self.weights):
                self.weights[cnt] = numpy.array([[utils.rand(-vmin, vmax) for i in range(w.shape[1])] for j in range(w.shape[0])])

    def get_signal_from_pre_unit(self, inputs):
        return utils.flatten([u.forward_propagation([inputs[i], ]).tolist() for u, i in zip(self.pre_units, self.input_index)])

    def forward_propagation(self, inputs, evaluate=0):
        if self.unit_type == "connected":
            inputs = self.get_signal_from_pre_unit(inputs)
        else:
            inputs = utils.flatten(inputs)
        inputs = numpy.append(inputs, [self.const])
        if len(inputs) != self.n_layers[0]:
            raise ValueError("wrong number of inputs")

        self.nodes[0].set_signal(inputs)
        for cnt, (n, n_next) in enumerate(zip(self.nodes, self.nodes[1:])):
            n_next.set_signal(n.forward_propagation(self.funcs[cnt][0], self.weights[cnt], evaluate=evaluate))
        return self.cost_func.func(self.nodes[-1].signal)

    def back_propagation(self, targets, epoch, **kwargs):
        """momentan SDG"""
        if "error" in kwargs.keys():
            error = kwargs["error"]
        else:
            error = self.cost_func.derv(self.nodes[-1].signal, targets)
        if len(error) != self.n_layers[-1]:
            print(len(error), self.n_layers[-1])
            raise ValueError("wrong number of target values")
        delta = 0
        buff = self.weights
        for cnt, (n, n_prev) in enumerate(zip(self.nodes[::-1], self.nodes[::-1][1:])):
            if cnt != 0: error = numpy.dot(self.weights[-cnt].T, delta)
            weight_delta, rs, delta = n_prev.back_propagation(error, self.funcs[-cnt-1][1], self.rs[-cnt-1],
                                                              self.beta, self.alpha, self.epsilon,
                                                              n.signal, self.gamma, self.weights[-cnt-1],
                                                              self.weights_buff[-cnt-1], epoch, cnt, n)
            self.weights[-cnt-1] += weight_delta
            self.rs[-cnt-1] = rs
            delta = delta
        self.weights_buff = buff
        error_in = numpy.dot(self.weights[0].T, delta)
        return self.cost_func.cost(self.nodes[-1].signal, targets, ), error_in



        # for n in range(len(self.signals[1:])):
        #     if n != 0: error = numpy.dot(self.weights[-n].T, delta)
        #     delta = numpy.array([f(i) for f, i in zip(self.funcs[-n-1][1], self.signals[-n-1])]) * error
        #     self.rs[-n-1] = self.beta * self.rs[-n-1] + (1 - self.beta) * (delta * delta).mean()
        #     self.weights[-n-1] += (self.alpha / (1 + self.epsilon * (epoch + (1+n)**-2)) * numpy.array([i * self.signals[-n-2] for i in delta / numpy.sqrt(self.rs[-n-1] + 1E-4)]) + self.gamma * (self.weights[-n-1] - self.weights_buff[-n-1]))
        # self.weights_buff = buff
        # error_in = numpy.dot(self.weights[0].T, delta)
        # return self.cost_func.cost(self.signals[-1], targets, ), error_in

    def evaluate(self, patterns, save=0):
        """"""
        cnt, corrct = 0, 0
        error = list()
        res = numpy.zeros(self.n_layers[-1] ** 2).reshape(self.n_layers[-1], self.n_layers[-1])
        out = list()
        # w = numpy.array([1 / 251., 1 / 781., 1 / 451.])
        for pat in patterns:
            p, a = pat
            ans = numpy.array([0] * len(a))
            ans = numpy.array(self.forward_propagation(p, evaluate=1))
            out.append(ans)
            error.append(self.cost_func.cost(self.nodes[-1].signal, a))
            res[list(a).index(max(a)), :] += ans
            cnt += 1
            corrct += int(list(a).index(max(a)) == list(ans).index(max(ans)))
        for i in range(self.n_layers[-1]):
            res[i, :] /= res[i, :].sum()
        print("-"*100)
        print("%d / %d, %d%%, error%.3f" % (corrct, cnt, round(corrct / float(cnt) * 100., 2), sum(error) / len(error)))
        print(pandas.DataFrame(res, columns=["pred%s" % i for i in range(self.n_layers[-1])], index=["corr%s" % i for i in range(self.n_layers[-1])]))
        print("-"*100)
        # self.score = round(corrct / float(cnt) * 100, 2)
        self.score_prvs = self.score
        self.score = sum(error) / len(error)
        return ans

    def save(self):
        now = datetime.datetime.now()
        if not os.path.exists("./pickle"):
            os.mkdir("./pickle")
        with open('./pickle/%s_gen%s_score%s_%s_%02d%02d_%02d%02d%02d' % (self.name, self.generation, str(self.score).replace(".", "p"), now.year, now.month, now.day, now.hour, now.minute, now.second), mode='wb') as f:
            pickle.dump(self, f)
            print("saved as % s" % f.name)

    def log(self, len_pat):
        f = open("log.txt", "a")
        f.write("%s " % self.name + " ".join(map(str, self.n_layers)))
        f.write(" %s %s %s %s %s " % (self.alpha, self.beta, self.gamma, self.delta, self.epsilon))
        f.write("%s %s %s " % (self.generation, len_pat, self.score))
        f.write(self.comment)
        f.write("\n")

    def check_progress(self, cnt, error, th=1e-7):
        error = list(error)[-len(error) // 2:]
        x = numpy.array(list(range(len(error))))
        y = numpy.array(list(error))
        a, _, _, _, _ = scipy.stats.linregress(x, y)
        if numpy.array(list(error)).std() < 0.000001:
            return StopIteration, ("saturated")
        if numpy.array(list(error)).mean() > 10:
            return StopIteration, ("overflow")
        if numpy.array(list(error)).std() != numpy.array(list(error)).std():
            return StopIteration, ("encontered nan")
        return 1, (cnt, numpy.array(error).mean(), a)

    def describe(self):
        print("-"*30)
        print("%s, gen %s" % (self.name, self.generation))
        print("size ->", "-".join(map(str, self.n_layers)))
        print("alpha, beta, gamma, delta, epsilon -> %-.5f, %-.2f, %-.2f, %-.5f, %-.2f" % (self.alpha, self.beta, self.gamma, self.delta, self.epsilon))
        print("evaluation -> %s" % self.cost_func.name)
        print("comment -> %s" % self.comment)

    def train(self, patterns, eval_fanc=0, arg=0, epoch=10, interval=1, save=0, fukusyu=0, pre_unit_train=0):
        """momentam SGD"""
        print("Start train..")
        self.describe()
        for i in range(epoch):
            error, pat_fukusyu = self.online_train(patterns, i, pre_unit_train=pre_unit_train)
            if fukusyu: self.online_train(pat_fukusyu, i, pre_unit_train=pre_unit_train)
        if eval_fanc:
            print("\n")
            yield i, error, eval_fanc(*arg)
        else:
            yield i, error
        self.generation += 1

        if save and self.score < 1.01 and self.score < self.score_prvs: self.save()
        self.log(len(patterns))

    def online_train(self, patterns, epoch, pre_unit_train = 0, report_freq=100):
        error = list()
        errors = dict()
        start = time.time()
        for _ in range(random.randint(1, 3)): random.shuffle(patterns)
        for cnt, p in enumerate(patterns):
            inputs = p[0]
            targets = p[1]
            self.forward_propagation(inputs)
            error_this, error_pre_units = self.back_propagation(targets, epoch)
            if pre_unit_train and self.unit_type == "connected":
                n_outs = [0, ]
                for u in self.pre_units: n_outs.extend([n_outs[-1] + u.n_layers[-1]])
                for u, i, j in zip(self.pre_units, n_outs, n_outs[1:]):
                    u.back_propagation(-1, epoch, error=error_pre_units[i: j])
            error.append(error_this)
            errors.update({error_this: p})
            if cnt % report_freq == 0 and cnt != 0:
                self.report(cnt, start, len(patterns), error, epoch)
        (_, patterns_fukusyu) = zip(*sorted(errors.items(), key=lambda x: x[0], reverse=1))
        return sum(error) / len(patterns), list(patterns_fukusyu[:len(patterns) // 3])

    def report(self, cnt, start, len_pat, error, epoch):
        ratio = cnt / float(len_pat)
        time_remain = int((time.time() - start) / ratio * (1 - ratio))
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        time_remain = "%d:%02d:%02d" % (h, m, s)
        flag, res = self.check_progress(cnt, error, 1e-6)
        try:
            if numpy.sign(res[2]) > 0: trend = "+"
            else: trend = "-"
        except: trend = "e"
        if flag == StopIteration:
            print(res)
            raise StopIteration
        else:
            print ("epech[%04d], cnt[%06d], cost[%6.4s], trend[%s], [%5.1f]%%, [%s]\r" % (epoch, cnt, res[1], trend, ratio*100, time_remain), end="")