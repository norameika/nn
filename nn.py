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
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

pandas.set_option('display.width', 1000)
seaborn.set(style="darkgrid", palette="muted", color_codes=True)


class prop(object):
    def __init__(self, kind, inp_shape, err_shape, *args, **kwargs):
        self.kind = kind
        self.inp = numpy.array([])
        self.back_proptable = 1
        self.inp_shape = inp_shape
        self.err_shape = err_shape

    def frwd_prop(self, inp, *args, **kwargs):
        self.inp = numpy.array(inp)
        pass

    def frwd_prop_eval(self, inp, *args, **kwargs):
        self.inp = numpy.array(inp)
        if self.kind == "node":
            return self.frwd_prop_eval(inp)
        else:
            return self.frwd_prop(inp)
        pass

    def back_prop(self, err, *args, **kwargs):
        pass

    def back_prop_out(self, tar, *args, **kwargs):
        pass

    def check_inp_dim(self, inp, *args, **kwargs):
        return len(inp.shape)

    def check_gain(self, bias=0):
        fwrd = numpy.array(self.frwd_prop((numpy.ones(numpy.prod(self.inp_shape)) + bias).reshape(self.inp_shape)))
        back = numpy.array(self.back_prop((numpy.ones(numpy.prod(self.err_shape)) + bias).reshape(self.err_shape)))
        print("gain - fwrd ->", "%1.9s" % (fwrd.ravel().sum() / numpy.prod(self.inp_shape)))
        print("gain - back ->", "%1.9s" % (back.ravel().sum() / numpy.prod(self.err_shape)))

    def set_hyper_params(self, params):
        self.alpha, self.beta, self.gamma, self.delta = params


class node(prop):
    """
    activation layer
    """
    def __init__(self, n, do=1):
        prop.__init__(self, "node", (n), (n))
        self.n = n
        self.mask = numpy.array([1] * n)
        self.funcs = [f for f in [random.choice([functions.relu, functions.tanh]) for i in range(self.n)]]
        # self.funcs = [f for f in [random.choice([functions.relu]) for i in range(self.n)]]
        self.do = do

    def set_activation(self, funcs):
        self.funcs = [f for f in [random.choice(funcs) for i in range(self.n)]]

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        if self.do: self.update_dropout_mask(self.n)
        out = numpy.array([f.func(j) for f, j in zip(self.funcs, self.mask * inp)])
        # out = out / out.sum() * inp.sum()
        return out.reshape([1] + list(out.shape))

    def frwd_prop_eval(self, inp):
        super().frwd_prop(inp)
        self.mask = numpy.array([1] * self.n)
        out = numpy.array([f.func(j) for f, j in zip(self.funcs, self.mask * inp)])
        # out = out / out.sum() * inp.sum()
        return out.reshape([1] + list(out.shape))

    def back_prop(self, err):
        out = numpy.array([f.derv(j) for f, j in zip(self.funcs, self.mask * self.inp)] * err)
        out = out / out.sum() * err.sum()
        return out.reshape([1] + list(out.shape))

    def update_dropout_mask(self, n):
        mask = numpy.array([i < 1. and i > -1. for i in numpy.random.normal(0, 1, n)]).astype(numpy.int16)
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
        self.alpha = 10 ** (-3 + numpy.random.normal(0, 1))
        self.beta = min(1, numpy.random.normal(0.8, 0.05))
        self.gamma = min(1, numpy.random.normal(0.8, 0.05))
        self.delta = 10 ** (-4 + numpy.random.normal(0, 1))
        self.r= numpy.array([10.] * self.n_out)
        self.initialization(0, self.delta)
        self.alpha = 1E-4
        self.beta = 0.9
        self.gamma = 0.9
        self.delta = 1E-5

    def initialization(self, *args):
        mean, sig = args
        self.weight = numpy.array([numpy.random.normal(mean, sig, self.weight.shape[1]) for i in range(self.weight.shape[0])])

    def frwd_prop(self, inp):
        inp = numpy.append(inp, [1])
        super().frwd_prop(inp)
        out = numpy.dot(self.weight, inp)
        self.attenuator(out)
        return out.reshape([1] + list(out.shape))

    def back_prop(self, err):
        buff = self.weight
        self.r = self.beta * self.r + (1 - self.beta) * (err * err).mean()
        self.weight += self.alpha * numpy.array([i * self.inp for i in err / numpy.sqrt(self.r + 1E-6)]) + self.gamma * (self.weight - self.weight_buff)
        self.weight_buff = buff
        out = numpy.dot(self.weight.T[:-1], err)
        self.attenuator(out)
        return out.reshape([1] + list(out.shape))

    def reset_buf(self):
        self.weight_buff = copy.deepcopy(self.weight)

    def attenuator(self, sig):
        a = abs(sig).max()
        if a > 10:
            self.weight[self.weight > abs(self.weight).max() / 2]
            self.weight[self.weight < -abs(self.weight).max() / 2] = 0
            self.weight_buff = copy.deepcopy(self.weight)
            self.alpha *= 0.5


class cnvl(prop):
    def __init__(self, ks, stride, inp_size, n_filters, zp=0, *args, **kwargs):
        self.ks = ks
        self.stride = stride
        self.zp = zp
        self.inp_size, self.inp_size_r = list(inp_size), list(inp_size)
        if self.zp: self.inp_size[0] += 2 * (self.ks); self.inp_size[1] += 2 * (self.ks)
        self.n_filters = n_filters

        self.x = numpy.arange(0, self.inp_size[1] - self.ks, self.stride)
        self.y = numpy.arange(0, self.inp_size[0] - self.ks, self.stride)
        self.feat_map_size = (len(self.y), len(self.x))
        prop.__init__(self, "cnvl", inp_size, [self.n_filters] + list(self.feat_map_size))

        self.out_shape = (self.feat_map_size[0], self.feat_map_size[1])
        self.alpha = 1E-4
        self.beta = 0.9
        self.delta = 0.1
        self.th = 1.
        self.moment = [list() for _ in range(n_filters)]
        self.k = [1. for _ in range(n_filters)]

        self.kernel = [self.gen_kernal() for _ in range(n_filters)]
        self.kernel_buf = copy.deepcopy(self.kernel)
        self.b = abs(numpy.random.normal(0., 1E-10, n_filters))

        self.out_slice = slice(0, None)
        if self.zp: slice(self.ks, -self.ks)

    def frwd_prop(self, inp):
        inp_raw = inp
        if self.zp: inp = self.zero_padding(inp)
        cnvled = numpy.array([self.crop(x, y, inp, self.ks, self.ks) for x, y in zip(*self.gen_strides())])
        cnvled = numpy.array([numpy.sum(cnvled * k[None, :] + b, axis=tuple(range(1, 1 + len(self.inp_size)))).reshape(*self.out_shape) for k, b in zip(self.kernel, self.b)])
        cnvled = [c / c.sum() * inp_raw.sum() for c in cnvled]
        self.inp = inp
        # self.attenuator_all(cnvled)
        self.cnvled = cnvled
        return cnvled

    def back_prop(self, errs):
        out = list()
        for cnt, (err, kernel) in enumerate(zip(errs, self.kernel)):
            arr_inp = numpy.array([self.crop(x, y, self.inp, self.feat_map_size[1], self.feat_map_size[0]) for x, y in zip(*self.gen_strides_back_prop_kernel())])
            if len(self.inp_size) == 3:
                delta = numpy.sum(err[None, :, :, None] * arr_inp, axis=(1, 2)).reshape((self.ks, self.ks, 3))
            else:
                delta = numpy.sum(err[None, :, :] * arr_inp, axis=(1, 2)).reshape((self.ks, self.ks))

            buf = self.kernel[cnt]
            self.kernel[cnt] += self.k[cnt] * self.alpha * delta + self.beta * (self.kernel[cnt] - self.kernel_buf[cnt])
            self.b[cnt] = self.k[cnt] * self.alpha * numpy.sum(err)
            self.kernel_buf[cnt] = buf
            self.moment[cnt].append(abs((self.kernel[cnt] - self.kernel_buf[cnt]).ravel()).sum())

            arr = self.pad(err)
            arr = numpy.array([self.crop(x, y, arr, self.ks, self.ks) for x, y in zip(*self.gen_strides_back_prop())])
            if len(self.inp_size) == 3:
                o = numpy.mean(arr[:, :, :, None] * self.flip(kernel)[None, :] / (self.ks), axis=tuple(range(1, 1 + len(self.inp_size)))).reshape(*self.inp_size[:2])[self.out_slice, self.out_slice, :]
            else:
                o = numpy.mean(arr * self.flip(kernel)[None, :] / (self.ks), axis=tuple(range(1, 1 + len(self.inp_size)))).reshape(*self.inp_size[:2])[self.out_slice, self.out_slice]
            if self.zp: o = o[self.ks: - self.ks, self.ks: - self.ks]
            out.append(o / o.sum() * err.sum())
            # self.attenuator(out[-1], cnt)
        self.kernel = numpy.array(self.kernel)
        return out

    def crop(self, x, y, arr, sizex, sizey):
        return arr[y: y+sizey, x: x+sizex]

    def pad(self, arr):
        arr = arr.reshape(self.feat_map_size)
        out = numpy.zeros((self.ks + self.inp_size[0], self.ks + self.inp_size[1]))
        out[slice(self.ks, arr.shape[0] * self.stride + self.ks, self.stride), slice(self.ks, arr.shape[1] * self.stride + self.ks, self.stride)] = arr
        return out

    def zero_padding(self, arr):
        out = numpy.zeros((self.inp_size[0], self.inp_size[1])).reshape(*self.inp_size[:2])
        out[self.ks: - self.ks, self.ks: - self.ks] = arr
        return out

    def gen_kernal(self):
        for cnt, i in enumerate(self.inp_size[2:]):
            if cnt == 0:
                kernel = [[numpy.random.normal(self.delta, self.delta, i) for _ in range(self.ks)] for _ in range(self.ks)]
                continue
            kernel = [kernel for _ in range(i)]
        if 'kernel' not in locals():
            kernel = [numpy.random.normal(self.delta, self.delta, self.ks) for _ in range(self.ks)]
        kernel = numpy.array(kernel)
        # kernel[kernel > 1] = 1
        return kernel

    def gen_strides(self):
        x = numpy.arange(0, self.inp_size[1] - self.ks, self.stride)
        y = numpy.arange(0, self.inp_size[0] - self.ks, self.stride)
        xx, yy = numpy.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def gen_strides_back_prop_kernel(self):
        x = numpy.arange(0, self.ks)
        y = numpy.arange(0, self.ks)
        xx, yy = numpy.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def gen_strides_back_prop(self):
        x = numpy.arange(1, self.inp_size[1] + 1)
        y = numpy.arange(1, self.inp_size[0] + 1)
        xx, yy = numpy.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def flip(self, arr):
        return numpy.fliplr(arr)[::-1]

    def show_kernel(self):
        for kernel in self.kernel:
            Image.fromarray((kernel - kernel.min()) / (kernel.max() - kernel.min()) * 255).show()

    def show_inp(self):
        Image.fromarray((self.inp - self.inp.min()) / (self.inp.max() - self.inp.min()) * 255).show()
        for cnv in self.cnvled:
            Image.fromarray((cnv - cnv.min()) / (cnv.max() - cnv.min()) * 255).show()

    def reset_buf(self):
        self.kernel_buf = copy.deepcopy(self.kernel)

    def attenuator(self, sig, cnt):
        a = abs(sig).max()
        if numpy.array(self.moment[cnt]).mean() < 1E-16 and len(self.moment[cnt]) > 1000:
            self.k[cnt] *= 2
            self.k[cnt] = min(self.k[cnt], 1)
            # print("updated k", self.k)
            self.moment[cnt] = list()
        if a > 10:
            self.kernel[cnt][self.kernel[cnt] > abs(self.kernel[cnt]).max() / 2] = 0
            self.kernel[cnt][self.kernel[cnt] < -abs(self.kernel[cnt]).max() / 2] = 0
            self.kernel_buf = copy.deepcopy(self.kernel)
            self.k[cnt] *= 0.5

    def attenuator_all(self, sig):
        a = abs(numpy.array(sig)).max()
        if a > 10:
            self.kernel[self.kernel > abs(self.kernel).max()/2] = 0
            self.kernel[self.kernel < -abs(self.kernel).max()/2] = 0


class maxpool(prop):
    def __init__(self, win_size, stride, inp_size):
        self.win_size = win_size
        self.stride = stride
        self.inp_size = inp_size
        self.index_map = numpy.array([])
        self.back_prop_map = numpy.zeros(numpy.prod(self.inp_size)).reshape(*self.inp_size)
        prop.__init__(self, "maxpool", inp_size, (self.inp_size[0] // self.stride, self.inp_size[1] // self.stride))

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        self.reset_back_prop_map()
        maxpool = numpy.array([self.crop(x, y, inp, self.win_size, self.win_size) for x, y in zip(*self.gen_strides())]).reshape([self.inp_size[0] // self.stride, self.inp_size[1] // self.stride]) * 4
        return maxpool.reshape([1] + list(maxpool.shape))

    def back_prop(self, err):
        self.back_prop_map[self.back_prop_map == 1] = err.ravel()
        return numpy.array([[i] for i in self.back_prop_map.reshape([1] + list(self.back_prop_map.shape))])

    def crop(self, x, y, arr, sizex, sizey):
        crop = arr[y: y+sizey, x: x+sizex]
        argmax = crop.argmax()
        index_local_map = numpy.unravel_index(argmax, [self.win_size, self.win_size])
        self.back_prop_map[index_local_map[0] + y, index_local_map[1] + x] = 1
        return crop.ravel()[argmax]

    def gen_strides(self):
        x = numpy.arange(0, self.inp_size[1], self.stride)
        y = numpy.arange(0, self.inp_size[0], self.stride)
        xx, yy = numpy.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())

    def reset_back_prop_map(self):
        self.back_prop_map = numpy.zeros(numpy.prod(self.inp_size)).reshape(*self.inp_size)


class deploy(prop):
    def __init__(self):
        prop.__init__(self, "deploy", (), ())
        pass

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        out = numpy.array(inp).ravel()
        return out.reshape([1] + list(out.shape))

    def back_prop(self, err):
        return numpy.array([[i] for i in numpy.array(err).reshape(self.inp.shape)])

    def gen_strides(self):
        x = numpy.arange(0, self.inp_size[1] - self.win_size, self.stride)
        y = numpy.arange(0, self.inp_size[0] - self.win_size, self.stride)
        xx, yy = numpy.meshgrid(x, y)
        return (xx.ravel(), yy.ravel())


class deploy_nested(prop):
    def __init__(self, length):
        prop.__init__(self, "deploy", (length,), (length,))
        self.node = node(length, do=0)
        pass

    def frwd_prop(self, inp):
        super().frwd_prop(inp)
        return self.node.frwd_prop(numpy.array(inp).ravel())[0].reshape(self.inp.shape)

    def back_prop(self, err):
        return numpy.array([[i] for i in self.node.back_prop(numpy.array(err).ravel())[0].reshape(self.inp.shape)])


class model(object):
    def __init__(self):
        self.prop = list()
        self.gen = 0
        self.score = 999
        self.name = "no name"
        self.errs = list()
        self.tar = 0

        # self.hyper_params = [10 ** (-3 + numpy.random.normal(0, 1)),
        #                      min(1, numpy.random.normal(0.8, 0.05)),
        #                      min(1, numpy.random.normal(0.8, 0.05)),
        #                      10 ** (-4 + numpy.random.normal(0, 1))]

        self.hyper_params = [0.00001,
                             0.9,
                             0.9,
                             1E-5]

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
        x = numpy.array(list(range(len(err))))
        y = numpy.array(list(err))
        a, _, _, _, _ = scipy.stats.linregress(x, y)
        if numpy.array(list(err)).std() < 0.0000000001:
            return StopIteration, ("saturated")
        if numpy.array(list(err)).mean() > 10:
            return StopIteration, ("overflow")
        if numpy.array(list(err)).std() != numpy.array(list(err)).std():
            return StopIteration, ("encontered nan")
        return 1, (cnt, numpy.array(err).mean(), a)

    def report(self, cnt, start, len_pat, err, epoch):
        ratio = cnt / float(len_pat)
        time_remain = int((time.time() - start) / ratio * (1 - ratio))
        m, s = divmod(time_remain, 60)
        h, m = divmod(m, 60)
        time_remain = "%d:%02d:%02d" % (h, m, s)
        flag, res = self.check_progress(cnt, err, 1e-6)
        try:
            if numpy.sign(res[2]) > 0: trend = "+"
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
        res = numpy.zeros(n_out ** 2).reshape(n_out, n_out)
        out = list()
        # w = numpy.array([1 / 251., 1 / 781., 1 / 451.])
        for pat in patterns:
            p, tar = pat
            ans = numpy.array([0] * len(tar))
            ans = numpy.array(self.frwd_prop_eval(p))
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
                print(p.kind, "rest buf")

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
        for pr in self.prop:
            prams = numpy.array(self.hyper_params * numpy.array([k, 1, 1, 1]))
            for p in pr:
                if p.kind == "cnvl":
                    p.set_hyper_params(prams)
                    p.__init__(p.ks, p.stride, p.inp_size_r, p.n_filters)
                if p.kind == "fc":
                    p.set_hyper_params(prams)
                    p.__init__(p.n_inp_r, p.n_out)
            # if p.kind == "cnvl" or p.kind == "fc":
            #     k *= numpy.exp(numpy.log(3.333) / l)

    def describe(self):
        for cnt, pr in enumerate(self.prop):
            print("%02dlayer" % cnt, ",".join([p.kind for p in pr]))


if __name__ == '__main__':
    a = numpy.array([1, 1])
    print(len(a.shape))