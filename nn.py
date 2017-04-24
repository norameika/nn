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
        self.inp_shape = inp_shape
        self.err_shape = err_shape
        self.inp = np.zeros(int(np.prod(self.inp_shape))).reshape(self.inp_shape)
        self.err = np.zeros(int(np.prod(self.err_shape))).reshape(self.err_shape)
        self.back_proptable = 1

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
        self.err = err
        pass

    def back_prop_out(self, tar, *args, **kwargs):
        pass

    def check_inp_dim(self, inp, *args, **kwargs):
        return len(inp.shape)

    def check_gain(self, bias=0):
        fwrd = np.array(self.frwd_prop((np.ones(np.prod(self.inp_shape)) + bias).reshape(self.inp_shape)))
        back = np.array(self.back_prop((np.ones(np.prod(self.err_shape)) + bias).reshape(self.err_shape)))
        print("gain - fwrd ->", "%1.9s" % (fwrd.ravel().sum() / np.prod(self.inp_shape)))
        print("gain - back ->", "%1.9s" % (back.ravel().sum() / np.prod(self.err_shape)))

    def set_hyper_params(self, params):
        self.alpha, self.beta, self.gamma, self.delta = params


class node(prop):
    """
    activation layer
    """
    def __init__(self, n, do=1):
        prop.__init__(self, "node", (n), (n))
        self.n = n
        self.mask = np.array([1] * n)
        self.funcs = [f for f in [random.choice([functions.relu, functions.tanh]) for i in range(self.n)]]
        self.do = do

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

        # hyper parameters
        self.alpha = 10 ** (-4 + np.random.normal(0, 1))
        self.beta = min(1, np.random.normal(0.8, 0.05))
        self.gamma = min(1, np.random.normal(0.8, 0.05))
        self.delta = 10 ** (-4 + np.random.normal(0, 1))
        # self.r = np.array([10.] * self.n_out)
        self.r = np.zeros(np.prod(self.weight.shape)).reshape(self.weight.shape) + 10.
        self.alpha = 1E-2
        self.beta = 0.9
        self.gamma = 0.9
        self.delta = 1E-6
        self.initialization(0., self.delta)

    def initialization(self, *args):
        mean, sig = args
        self.weight = np.array([np.random.normal(mean, sig, self.weight.shape[1]) for i in range(self.weight.shape[0])])

    def reinit(self):
        self.weight = np.array([np.random.normal(0, self.delta, self.weight.shape[1]) for i in range(self.weight.shape[0])])

    def frwd_prop(self, inp):
        inp = np.append(inp, [1])
        super().frwd_prop(inp)
        out = np.dot(self.weight, inp)
        return out.reshape([1] + list(out.shape)) 

    def back_prop(self, err):
        super().back_prop(err)
        delta = self.inp * err[:, None]
        buff = self.weight
        self.r = self.beta * self.r + (1 - self.beta) * (delta * delta)
        self.weight += self.alpha / (np.sqrt(self.r) + 1E-4) + self.gamma * (self.weight - self.weight_buff)
        self.weight_buff = buff
        out = np.dot(self.weight.T[:-1], err)
        return out.reshape([1] + list(out.shape))

    def reset_buf(self):
        self.weight_buff = copy.deepcopy(self.weight)


class cnvl(prop):
    def __init__(self, ks, stride, inp_size, n_filters, zp=0, *args, **kwargs):
        self.ks = ks
        self.stride = stride
        self.zp = zp
        self.inp_size, self.inp_size_r = list(inp_size), list(inp_size)
        if self.zp:
            self.inp_size[0] = self.inp_size[0] + 2 * self.zp
            self.inp_size[1] = self.inp_size[1] + 2 * self.zp
        self.n_filters = n_filters

        self.x = np.arange(0, self.inp_size[1] - self.ks + 1, self.stride)
        self.y = np.arange(0, self.inp_size[0] - self.ks + 1, self.stride)
        self.feat_map_size = (len(self.y), len(self.x))
        prop.__init__(self, "cnvl", inp_size, [self.n_filters] + list(self.feat_map_size))

        self.out_shape = (self.feat_map_size[0], self.feat_map_size[1])
        self.alpha = 1E-2
        self.beta = 0.9
        self.gamma = 0.9
        self.delta = 1E-6
        self.th = 1.

        self.kernel = [self.gen_kernal() for _ in range(n_filters)]
        self.kernel_buf = copy.deepcopy(self.kernel)
        self.b = abs(np.random.normal(0., 1E-10, n_filters))
        self.r = [np.zeros(self.ks * self.ks).reshape((self.ks, self.ks)) + 10. for _ in range(n_filters)]

        self.out_slice = slice(0, None)
        if self.zp:
            self.out_slice = slice(self.zp, -self.zp)

    def reinit(self):
        self.kernel = [self.gen_kernal() for _ in range(self.n_filters)]
        self.kernel_buf = copy.deepcopy(self.kernel)

    def frwd_prop(self, inp):
        inp_raw = inp
        if self.zp: inp = self.zero_padding(inp)
        cnvled = np.array([self.crop(x, y, inp, self.ks, self.ks) for x, y in zip(*self.gen_strides())])
        cnvled = np.array([np.sum(cnvled * k[None, :] + b, axis=tuple(range(1, 1 + len(self.inp_size)))).reshape(self.out_shape) for k, b in zip(self.kernel, self.b)])
        self.inp = inp
        self.cnvled = cnvled
        return cnvled

    def back_prop(self, errs):
        out = list()
        for cnt, (err, kernel) in enumerate(zip(errs, self.kernel)):
            arr_inp = np.array([self.crop(x, y, self.inp, self.feat_map_size[1], self.feat_map_size[0]) for x, y in zip(*self.gen_strides_back_prop_kernel())])
            if len(self.inp_size) == 3:
                delta = np.sum(err[None, :, :, None] * arr_inp, axis=(1, 2)).reshape((self.ks, self.ks, 3))
            else:
                delta = np.sum(err[None, :, :] * arr_inp, axis=(1, 2)).reshape((self.ks, self.ks))

            self.r[cnt] = self.beta * self.r[cnt] + (1 - self.beta) * (delta * delta)
            buf = self.kernel[cnt]
            self.kernel[cnt] += self.alpha / (np.sqrt(self.r[cnt]) + 1E-4) * delta + self.gamma * (self.kernel[cnt] - self.kernel_buf[cnt])
            self.b[cnt] = self.alpha * np.sum(err)
            self.kernel_buf[cnt] = buf

            arr = self.pad(err)
            arr = np.array([self.crop(x, y, arr, self.ks, self.ks) for x, y in zip(*self.gen_strides_back_prop())])
            if len(self.inp_size) == 3:
                o = np.mean(arr[:, :, :, None] * self.flip(kernel)[None, :], axis=tuple(range(1, 1 + len(self.inp_size)))).reshape(*self.inp_size[:2])[self.out_slice, self.out_slice, :]
            else:
                o = np.mean(arr * self.flip(kernel)[None, :], axis=tuple(range(1, 1 + len(self.inp_size)))).reshape(*self.inp_size[:2])[self.out_slice, self.out_slice]
            out.append(o)
        self.kernel = np.array(self.kernel)
        return out

    def crop(self, x, y, arr, sizex, sizey):
        return arr[y: y+sizey, x: x+sizex]

    def pad(self, arr):
        arr = arr.reshape(self.feat_map_size)
        out = np.zeros((self.ks * 2 + self.inp_size[0], self.ks * 2 + self.inp_size[1]))
        out[slice(self.ks, arr.shape[0] * self.stride + self.ks, self.stride), slice(self.ks, arr.shape[1] * self.stride + self.ks, self.stride)] = arr
        return out

    def zero_padding(self, arr):
        out = np.zeros((self.inp_size[0], self.inp_size[1])).reshape(*self.inp_size[:2])
        out[self.zp: - self.zp, self.zp: - self.zp] = arr
        return out

    def gen_kernal(self):
        for cnt, i in enumerate(self.inp_size[2:]):
            if cnt == 0:
                kernel = [[np.random.normal(0, self.delta, i) for _ in range(self.ks)] for _ in range(self.ks)]
                continue
            kernel = [kernel for _ in range(i)]
        if 'kernel' not in locals():
            kernel = [np.random.normal(0, self.delta, self.ks) for _ in range(self.ks)]
        kernel = np.array(kernel)
        # kernel[kernel > 1] = 1
        return kernel

    def gen_strides(self):
        x = np.arange(0, self.inp_size[1] - self.ks + 1, self.stride)
        y = np.arange(0, self.inp_size[0] - self.ks + 1, self.stride)
        xx, yy = np.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def gen_strides_back_prop_kernel(self):
        x = np.arange(0, self.ks)
        y = np.arange(0, self.ks)
        xx, yy = np.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def gen_strides_back_prop(self):
        x = np.arange(1, self.inp_size[1] + 1)
        y = np.arange(1, self.inp_size[0] + 1)
        xx, yy = np.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def flip(self, arr):
        return np.fliplr(arr)[::-1]

    def show_kernel(self):
        for kernel in self.kernel:
            Image.fromarray((kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255).show()

    def show_inp(self):
        Image.fromarray((self.inp - self.inp.min()) / (self.inp.max() - self.inp.min()) * 255).show()
        for cnv in self.cnvled:
            Image.fromarray((cnv - cnv.min()) / (cnv.max() - cnv.min()) * 255).show()

    def reset_buf(self):
        self.kernel_buf = copy.deepcopy(self.kernel)


class maxpool(prop):
    def __init__(self, win_size, stride, inp_size):
        self.win_size = win_size
        self.stride = stride
        self.inp_size = inp_size
        self.index_map = np.array([])
        self.back_prop_map = np.zeros(np.prod(self.inp_size)).reshape(*self.inp_size)
        prop.__init__(self, "maxpool", inp_size, (self.inp_size[0] // self.stride, self.inp_size[1] // self.stride))

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        self.reset_back_prop_map()
        maxpool = np.array([self.crop(x, y, inp, self.win_size, self.win_size) for x, y in zip(*self.gen_strides())]).reshape([self.inp_size[0] // self.stride, self.inp_size[1] // self.stride])
        return maxpool.reshape([1] + list(maxpool.shape))

    def back_prop(self, err):
        self.back_prop_map[self.back_prop_map == 1] = err.ravel()
        return np.array([[i] for i in self.back_prop_map.reshape([1] + list(self.back_prop_map.shape))])

    def crop(self, x, y, arr, sizex, sizey):
        crop = arr[y: y+sizey, x: x+sizex]
        argmax = crop.argmax()
        index_local_map = np.unravel_index(argmax, [self.win_size, self.win_size])
        self.back_prop_map[index_local_map[0] + y, index_local_map[1] + x] = 1
        return crop.ravel()[argmax]

    def gen_strides(self):
        x = np.arange(0, self.inp_size[1], self.stride)
        y = np.arange(0, self.inp_size[0], self.stride)
        xx, yy = np.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def reset_back_prop_map(self):
        self.back_prop_map = np.zeros(np.prod(self.inp_size)).reshape(*self.inp_size)


class deploy(prop):
    def __init__(self):
        prop.__init__(self, "deploy", (), ())
        pass

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        out = np.array(inp).ravel()
        return out.reshape([1] + list(out.shape))

    def back_prop(self, err):
        return np.array([[i] for i in np.array(err).reshape(self.inp.shape)])

    def gen_strides(self):
        x = np.arange(0, self.inp_size[1] - self.win_size, self.stride)
        y = np.arange(0, self.inp_size[0] - self.win_size, self.stride)
        xx, yy = np.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())


class deploy_nested(prop):
    def __init__(self, length):
        prop.__init__(self, "deploy", (length,), (length,))
        self.node = node(length, do=1)
        pass

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        return self.node.frwd_prop(np.array(inp).ravel())[0].reshape(self.inp.shape)

    def back_prop(self, err):
        return np.array([[i] for i in self.node.back_prop(np.array(err).ravel())[0].reshape(self.inp.shape)])


class model(object):
    def __init__(self):
        self.prop = list()
        self.gen = 0
        self.score = 999
        self.name = "no name"
        self.errs = list()
        self.tar = 0

        # self.hyper_params = [10 ** (-3 + np.random.normal(0, 1)),
        #                      min(1, np.random.normal(0.8, 0.05)),
        #                      min(1, np.random.normal(0.8, 0.05)),
        #                      10 ** (-4 + np.random.normal(0, 1))]

        self.hyper_params = {"fc":
                             [1E-3,
                              0.99,
                              0.99,
                              1E-6],
                             "cnvl":
                             [1E-3,
                              0.99,
                              0.99,
                              1E-6]}

    def add_prop(self, props):
        self.prop.append(props)

    def frwd_prop(self, inps):
        for cnt, props in enumerate(self.prop):
            inps_n = list()
            if props[0].kind == "deploy":
                inps = props[0].frwd_prop(inps)
            else:
                for p, inp in zip(props, inps):
                    inps_n.extend(p.frwd_prop(inp))
                inps = inps_n
        return inps

    def frwd_prop_eval(self, inps):
        for props in self.prop:
            inps_n = list()
            if props[0].kind == "deploy":
                inps = props[0].frwd_prop(inps)
            else:
                for p, inp in zip(props, inps):
                    inps_n.extend(p.frwd_prop_eval(inp))
                inps = inps_n
        return inps

    def back_prop(self, tar):
        self.tar = list(tar).index(1)
        self.errs = [list() for _ in range(len(self.prop)-1)]
        errs, cost = self.prop[-1][0].back_prop(tar)
        for cnt, props in enumerate(self.prop[::-1][1:]):
            self.errs[cnt] = [abs(e).mean() for e in errs]
            if props[0].kind == "deploy":
                errs = props[0].back_prop(errs)
            else:
                errs_n = list()
                for p, err in zip(props, errs):
                    if p.back_proptable: errs_n.extend(p.back_prop(err))
                errs = errs_n
        return cost

    def train(self, pats, pats_e, epoch, freq=1000):
        for _ in range(3): random.shuffle(pats)
        flag = 0
        errs = list()
        start = time.time()
        for ep in range(epoch):
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
            self.reset_buf()
            self.gen += 1
        return sum(errs) / len(pats)

    def train_for_animation(self, pats, pats_e, epoch, freq=1000):
        for _ in range(3): random.shuffle(pats)
        flag = 0
        errs = list()
        start = time.time()
        for ep in range(epoch):
            for cnt, p in enumerate(pats):
                inps = p[0]
                tar = p[1]
                self.frwd_prop(inps)
                errs.append(self.back_prop(tar))
                if cnt % freq == 0 and cnt != 0:
                    yield self.errs, self.tar
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
            self.reset_buf()
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
        # w = np.array([1 / 251., 1 / 781., 1 / 451.])
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
            # print(p, ans, tar)
        for i in range(n_out):
            res[i, :] /= res[i, :].sum()
        print("-"*100)
        print("%d / %d, %d%%, error%.3f" % (corr, cnt, round(corr / float(cnt) * 100., 2), sum(err) / len(err)))
        print(pandas.DataFrame(res, columns=["pred%s" % i for i in range(n_out)], index=["corr%s" % i for i in range(n_out)]))
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

    def reset_buf(self):
        for p in utils.flatten(self.prop):
            if "reset_buf" in dir(p):
                p.reset_buf()

    def show_kernel(self):
        for p in utils.flatten(self.prop):
            if "show_kernel" in dir(p):
                p.show_kernel()

    def show_inp(self):
        for p in utils.flatten(self.prop):
            if "show_inp" in dir(p):
                p.show_inp()

    def set_params(self):
        k = 1
        l = sum([1 if p[0].kind == "cnvl" or p[0].kind == "fc" else 0 for p in self.prop])
        for p in utils.flatten(self.prop):
            for kind, params in self.hyper_params.items():
                if p.kind == kind:
                    p.set_hyper_params(params)
                    p.reinit()

    def describe(self):
        for cnt, pr in enumerate(self.prop):
            print("%02dlayer" % cnt, ",".join([p.kind for p in pr]))


if __name__ == '__main__':
    a = np.array([1, 1])
    print(len(a.shape))