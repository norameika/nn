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

seaborn.set(style="darkgrid", palette="muted", color_codes=True)


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
            self.signals.append(numpy.array([0.] * n))

        # hyper parameters
        self.alpha = 10 ** (-3 + numpy.random.normal(0, 1))
        self.beta = min(1, numpy.random.normal(0.8, 0.05))
        self.gamma = min(1, numpy.random.normal(0.8, 0.05))
        self.delta = 10 ** (-3 + numpy.random.normal(0, 1))
        self.epsilon = 0.01

        self.comment = "no comment"

        self.rs = list()
        for n in self.n_layers[1:]:
            self.rs.append(numpy.array([10.] * n))

        self.name = "no name"
        self.const = 1
        self.generation = 0
        self.score = 0

        # default settnig
        self.funcs = list()
        self.set_activation_func([functions.tanh, functions.relu])
        self.cost_func = functions.square_error
        self.initialization("gaussian", 0, self.delta)

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

    def forward_propagation(self, inputs, dropout=0):
        inputs = numpy.append(inputs, [self.const])
        if len(inputs) != self.n_layers[0]:
            raise ValueError("wrong number of inputs")
        self.signals[0] = inputs
        for i in range(len(self.signals[1:])):
            self.signals[i + 1] = numpy.array([f(j) for f, j in zip(self.funcs[i][0], numpy.dot(self.weights[i], self.signals[i]))])
        return self.cost_func.func(self.signals[-1])

    def back_propagation(self, targets, epoch):
        """momentan SDG"""
        if len(targets) != self.n_layers[-1]:
            raise ValueError("wrong number of target values")
        error = self.cost_func.derv(self.signals[-1], targets)
        delta = 0
        buff = self.weights
        for n in range(len(self.signals[1:])):
            if n != 0: error = numpy.dot(self.weights[-n].T, delta)
            delta = numpy.array([f(i) for f, i in zip(self.funcs[-n-1][1], self.signals[-n-1])]) * error
            self.rs[-n-1] = self.beta * self.rs[-n-1] + (1 - self.beta) * (delta * delta).mean()
            self.weights[-n-1] += (self.alpha / (1 + self.epsilon * (epoch + (1+n)**-2)) * numpy.array([i * self.signals[-n-2] for i in delta / numpy.sqrt(self.rs[-n-1] + 1E-4)]) + self.gamma * (self.weights[-n-1] - self.weights_buff[-n-1]))
        self.weights_buff = buff
        error_in = numpy.dot(self.weights[0].T, delta)
        return self.cost_func.cost(self.signals[-1], targets, ), error_in

    def evaluate(self, patterns, save=0):
        """"""
        cnt, corrct = 0, 0
        res = numpy.zeros(self.n_layers[-1] ** 2).reshape(self.n_layers[-1], self.n_layers[-1])
        for p in patterns:
            ans = self.forward_propagation(p[0], dropout=0)
            corrct += self.evaluator(ans, p[1])
            res[list(p[1]).index(max(p[1])), list(ans).index(max(ans))] += 1
            cnt += 1
        print("%d / %d, raito%s" % (corrct, cnt, round(corrct / float(cnt) * 100, 2)))
        print(pandas.DataFrame(res, columns=["pred%s" % i for i in range(self.n_layers[-1])], index=["corr%s" % i for i in range(self.n_layers[-1])]))
        print("-"*30)
        self.score = round(corrct / float(cnt) * 100, 2)
        if save:
            self.save()
        return corrct / float(cnt)

    def evaluator(self, res, tar):
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0

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

    def clone(self, fp):
        with open(fp, mode='rb') as f:
            ob = pickle.load(f)
            self.n_layers = ob.n_layers
            self.signals = ob.signals
            self.name = ob.name
            self.weights = ob.weights
            self.weights_buff = ob.weights_buff
            self.funcs = ob.funcs
            self.rs = ob.rs
            self.generation = ob.generation + 1
            self.score = ob.score
            self.alpha, self.beta, self.gamma, self.delta, self.epsilon = ob.alpha, ob.beta, ob.gamma, ob.delta, ob.epsilon
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
        new_unit.name = unit.name + "_jr"
        return new_unit

    def check_progress(self, cnt, error, th=1e-7):
        x = numpy.array(list(range(len(error))))
        y = numpy.array(list(error))
        a, _, _, _, _ = scipy.stats.linregress(x, y)
        if numpy.array(list(error)).std() < 0.0001:
            return StopIteration, ("saturated")
        if numpy.array(list(error)).mean() > 10:
            return StopIteration, ("overflow")
        if numpy.array(list(error)).std() != numpy.array(list(error)).std():
            return StopIteration, ("encontered nan")
        return 1, (cnt, numpy.array(list(error)).mean(), a)

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
        print("-"*30)
        print("%s, gen %s" % (self.name, self.generation))
        print("size ->", "-".join(map(str, self.n_layers)))
        print("alpha, beta, gamma, delta, epsilon -> %-.5f, %-.2f, %-.2f, %-.5f, %-.2f" % (self.alpha, self.beta, self.gamma, self.delta, self.epsilon))
        print("evaluation -> %s" % self.cost_func.name)
        print("comment -> %s" % self.comment)

    def train(self, patterns, eval_fanc=0, arg=0, epoch=10, interval=1, save=0, fukusyu=0):
        """momentam SGD"""
        print("Start train..")
        self.describe()
        for i in range(epoch):
            error, pat_fukusyu = self.online_train(patterns, i)
            if fukusyu: self.online_train(pat_fukusyu, i)
            if i % interval == 0:
                if eval_fanc:
                    print("\n")
                    yield i, error, eval_fanc(*arg)
                else:
                    yield i, error
            self.generation += 1

        if save: self.save()
        self.log(len(patterns))

    def online_train(self, patterns, epoch, report_freq=1000):
        error = list()
        errors = dict()
        for cnt, p in enumerate(patterns):
            inputs = p[0]
            targets = p[1]
            self.forward_propagation(inputs)
            error_this, _ = self.back_propagation(targets, epoch)
            error.append(error_this)
            errors.update({error_this: p})
            if cnt % report_freq == 0 and cnt != 0:
                flag, res = self.check_progress(cnt, error, 1e-6)
                if flag == StopIteration:
                    print(res)
                    raise StopIteration
                else:
                    print ("epech[%04d], cnt[%06d], cost[%6.4s], trend[%s]\r" % (epoch, cnt, res[1], numpy.sign(res[2])), end="")
        (_, patterns_fukusyu) = zip(*sorted(errors.items(), key=lambda x: x[0], reverse=1))
        return sum(error) / len(patterns), patterns_fukusyu[:len(patterns) // 3]


class unity(object):
    def __init__(self, units, n_out):
        self.units = units
        self.n_out = n_out
        self.n_interm = sum([u.n_layers[-1] for u in self.units])
        self.n_layers = [self.n_interm, self.n_out]
        self.hyper_unit = unit(self.n_interm, n_out)
        self.name = "no name"
        self.comment = "no comment"
        self.hyper_unit.name = self.name
        self.hyper_unit.comment = self.comment

        self.score = -1
        self.generation = 10000

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
        f.write(" %s %s %s %s %s " % (self.hyper_unit.alpha, self.hyper_unit.beta, self.hyper_unit.gamma, self.hyper_unit.delta, self.hyper_unit.epsilon))
        f.write("%s %s %s " % (self.generation, len_pat, self.score))
        f.write(self.comment)
        f.write("\n")

    def set_activation_func(self, funcs, **kwargs):
        self.hyper_unit.set_activation_func(funcs, kwargs)

    def set_cost_func(self, cost_func):
        self.hyper_unit.cost_func = cost_func

    def get_interm_signal(self, inputs):
        return utils.flatten([u.forward_propagation(inputs).tolist() for u in self.units])

    def forward_propagation(self, inputs):
        return self.hyper_unit.forward_propagation(self.get_interm_signal(inputs))

    def back_propagation(self, targets, epoch):
        self.hyper_unit(targets, epoch)

    def train(self, patterns, eval_fanc=0, arg=0, epoch=10, interval=1, save=0, fukusyu=0):
        patterns = [[self.get_interm_signal(p[0]), p[1]] for p in patterns]
        for cnt, _ in enumerate(self.hyper_unit.train(patterns, eval_fanc=eval_fanc, arg=arg, epoch=epoch, interval=interval, save=0, fukusyu=fukusyu)):
            yield _
        self.generation += 1
        if save: self.save()
        self.log(len(patterns))

    def evaluate(self, patterns, save=0):
        """re-write by cost function"""
        cnt, corrct = 0, 0
        res = numpy.zeros(self.n_layers[-1] ** 2).reshape(self.n_layers[-1], self.n_layers[-1])
        for p in patterns:
            ans = self.forward_propagation(p[0])
            corrct += self.evaluator(ans, p[1])
            res[list(p[1]).index(max(p[1])), list(ans).index(max(ans))] += 1
            cnt += 1
        print("%d / %d, raito%s" % (corrct, cnt, round(corrct / float(cnt) * 100, 2)))
        print(pandas.DataFrame(res, columns=["pred%s" % i for i in range(self.n_layers[-1])], index=["corr%s" % i for i in range(self.n_layers[-1])]))
        print("-"*30)
        self.score = round(corrct / float(cnt) * 100, 2)
        return corrct / float(cnt)

    def evaluator(self, res, tar):
        """re-write by cost function"""
        if list(res).index(res.max()) == list(tar).index(1):
            return 1
        else:
            return 0


if __name__ == '__main__':
    pass
