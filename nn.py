# -*- coding: utf-8 -*-
import numpy as np
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
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

pandas.set_option('display.width', 1000)
seaborn.set(style="darkgrid", palette="muted", color_codes=True)


class prop(object):
    def __init__(self, kind, inp_shape, err_shape, *args, **kwargs):
        self.kind = kind
        self.back_proptable = 1
        self.inp_shape = inp_shape
        self.err_shape = err_shape
        self.inp = np.zeros(int(np.prod(self.inp_shape))).reshape(self.inp_shape)
        self.err = 0

    def frwd_prop(self, inp, *args, **kwargs):
        self.inp = np.array(inp)
        pass

    def frwd_prop_eval(self, inp, *args, **kwargs):
        self.inp = np.array(inp)
        if self.kind == "node":
            return self.frwd_prop_eval(inp)
        else:
            return self.frwd_prop(inp)
        pass

    def back_prop(self, err, *args, **kwargs):
        pass


class node(prop):
    """
    activation layer
    """
    def __init__(self, n, do=1):
        prop.__init__(self, "node", (n), (n))
        self.n = n
        self.mask = np.array([1] * n)
        self.funcs = [f for f in [random.choice([functions.relu, ]) for i in range(self.n)]]
        self.do = do  # dropout

    def set_activation(self, funcs):
        self.funcs = [f for f in [random.choice(funcs) for i in range(self.n)]]

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        if self.do: self.update_dropout_mask(self.n)
        out = np.array([f.func(j) for f, j in zip(self.funcs, self.mask * inp)])
        return out.reshape([1] + list(out.shape))

    def frwd_prop_eval(self, inp):
        super().frwd_prop(inp)
        self.mask = np.array([1] * self.n)
        out = np.array([f.func(j) for f, j in zip(self.funcs, self.mask * inp)])
        return out.reshape([1] + list(out.shape))

    def back_prop(self, err):
        super().back_prop(err)
        out = np.array([f.derv(j) for f, j in zip(self.funcs, self.mask * self.inp)] * err)
        return out.reshape([1] + list(out.shape))

    def update_dropout_mask(self, n):
        mask = np.array([i < 1. and i > -1. for i in np.random.normal(0, 1, n)]).astype(np.int16)
        self.mask = mask / 0.6825


class node_out(prop):
    """
    output layer
    """
    def __init__(self, n):
        prop.__init__(self, "node_out", (n), (n))
        self.n = n
        self.func = functions.logloss

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        return self.func.func(inp)

    def back_prop(self, tar):
        out = self.func.derv(self.inp, tar)
        return out.reshape([1] + list(out.shape)), self.func.cost(self.inp, tar)


class fc(prop):
    """
    fully connected layer
    """
    def __init__(self, n_inp, n_out, *args, **kwargs):
        prop.__init__(self, "fc", n_inp, n_out)
        self.n_inp, self.n_inp_r = n_inp + 1, n_inp  # fisrt layer for constant
        self.n_out = n_out

        self.weight = utils.gen_matrix(self.n_out, self.n_inp)
        self.weight_buff = copy.deepcopy(self.weight)

        self.initialization(0, 1 / self.n_inp / self.n_out / 100.)

    def initialization(self, *args):
        mean, sig = args
        self.weight = np.random.normal(mean, sig, self.weight.shape)

    def frwd_prop(self, inp):
        inp = np.append(inp, [1])
        super().frwd_prop(inp)
        out = np.dot(self.weight, inp)
        return out.reshape([1] + list(out.shape))

    def back_prop(self, err):
        super().back_prop(err)
        delta = self.inp * err[:, None]
        self.udpate_weight(delta)
        out = np.dot(self.weight.T[:-1], err)
        return out.reshape([1] + list(out.shape))

    def udpate_weight(self, delta, alpha=0.001):
        self.weight += - alpha * delta

    def reset_buf(self):
        self.weight_buff = copy.deepcopy(self.weight)


class model(object):
    def __init__(self):
        self.prop = list()
        self.gen = 0
        self.score = 999
        self.name = "no name"
        self.errs = list()

    def add_prop(self, props):
        self.prop.append(props)

    def frwd_prop(self, inps):
        for cnt, props in enumerate(self.prop):
            inps_n = list()
            for p, inp in zip(props, inps):
                inps_n.extend(p.frwd_prop(inp))
            inps = inps_n
        return inps

    def frwd_prop_eval(self, inps):
        for props in self.prop:
            inps_n = list()
            for p, inp in zip(props, inps):
                inps_n.extend(p.frwd_prop_eval(inp))
            inps = inps_n
        return inps

    def back_prop(self, tar):
        self.errs = [list() for _ in range(len(self.prop)-1)]
        errs, cost = self.prop[-1][0].back_prop(tar)
        for cnt, props in enumerate(self.prop[::-1][1:]):
            self.errs[cnt] = [abs(e).mean() for e in errs]
            errs_n = list()
            for p, err in zip(props, errs):
                if p.back_proptable: errs_n.extend(p.back_prop(err))
                else: return cost
            errs = errs_n
        return cost

    def train(self, pats, pats_e, epoch, fukusyu=0, freq=1000):
        flag = 0
        errs = list()
        start = time.time()
        for ep in range(epoch):
            for _ in range(3): random.shuffle(pats)
            for cnt, p in enumerate(pats):
                inps = p[0]
                tar = p[1]
                self.frwd_prop(inps)
                errs.append(self.back_prop(tar))
                if cnt % freq == 0 and cnt != 0:
                    # yield self.errs, self.tar
                    self.errs = list()
                    if self.report(cnt, start, len(pats), errs, ep) == StopIteration:
                        print("StopIteration")
                        flag = 1
                        break
            if flag: break

            for _ in range(3): random.shuffle(pats_e)
            try:
                self.eval(pats_e)
                self.save()
            except:
                pass
            self.gen += 1
        return sum(errs) / len(pats)

    def check_progress(self, cnt, err, th=1e-7):
        err = list(err)[-len(err) // 3:]
        x = np.array(list(range(len(err))))
        y = np.array(list(err))
        a, _, _, _, _ = scipy.stats.linregress(x, y)
        if np.array(list(err)).std() < 0.0000000001:
            return StopIteration, ("saturated")
        if np.array(list(err)).mean() > 10:
            return StopIteration, ("overflow")
        if np.array(list(err)).std() != np.array(list(err)).std():
            return StopIteration, ("encontered nan")
        return 1, (cnt, np.array(err).mean(), a)

    def report(self, cnt, start, len_pat, err, epoch):
        ratio = cnt / float(len_pat)
        time_remain = int((time.time() - start) / ratio * (1 - ratio))
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        time_remain = "%d:%02d:%02d" % (h, m, s)
        flag, res = self.check_progress(cnt, err, 1e-6)
        try:
            if np.sign(res[2]) > 0: trend = "+"
            else: trend = "-"
        except: trend = "e"
        if flag == StopIteration:
            print(res)
            return StopIteration
        else:
            print ("epech[%04d], cnt[%06d], cost[%6.4s], trend[%s], [%5.1f]%%, [%s]\r" % (epoch, cnt, res[1], trend, ratio*100, time_remain), end="")

    def eval(self, patterns):
        """"""
        cnt, corr = 0, 0
        err = list()
        n_out = self.prop[-1][0].n
        res = np.zeros(n_out ** 2).reshape(n_out, n_out)
        out = list()
        for pat in patterns:
            p, tar = pat
            ans = np.array([0] * len(tar))
            ans = np.array(self.frwd_prop_eval(p))
            out.append(ans)
            _, cost = self.prop[-1][0].back_prop(tar)
            err.append(cost)
            res[list(tar).index(max(tar)), :] += ans
            cnt += 1
            corr += int(list(tar).index(max(tar)) == list(ans).index(max(ans)))
        for i in range(n_out):
            res[i, :] /= res[i, :].sum()
        print("-"*100)
        print("%d / %d, %d%%, error%.3f" % (corr, cnt, round(corr / float(cnt) * 100., 2), sum(err) / len(err)))
        print(pandas.DataFrame(res, columns=["prdict%s" % i for i in range(n_out)], index=["crr ans%s" % i for i in range(n_out)]))
        print("-"*100)
        self.score = sum(err) / len(err)
        return ans

    def save(self):
        now = datetime.datetime.now()
        if not os.path.exists("./pickle"):
            os.mkdir("./pickle")
        with open('./pickle/%s_gen%s_score%s_%s_%02d%02d_%02d%02d%02d' % (self.name, self.gen, str(self.score).replace(".", "p"), now.year, now.month, now.day, now.hour, now.minute, now.second), mode='wb') as f:
            pickle.dump(self, f)
            print("saved as % s" % f.name)

    def show_kernel(self):
        for p in utils.flatten(self.prop):
            if "show_kernel" in dir(p):
                p.show_kernel()

    def show_inp(self):
        for p in utils.flatten(self.prop):
            if "show_inp" in dir(p):
                p.show_inp()

    def set_prms(self):
        for p in utils.flatten(self.prop):
            for kind, prms in self.hyper_prms.items():
                if p.kind == kind:
                    p.set_prms(prms)
                    p.reinit()

    def describe(self):
        for cnt, pr in enumerate(self.prop):
            print("%02dlayer" % cnt, ",".join([p.kind for p in pr]))


if __name__ == '__main__':
    pass