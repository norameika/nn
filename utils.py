import numpy as np
import random
try:
    from . import functions
except:
    import functions
from matplotlib import pylab as plt
import matplotlib.animation as animation
import pickle
import os
import re


def flatten(arr):
    x = []
    for s in arr:
        x.extend(s)
    return x


def rand(a, b):
    return (b - a) * random.random() + a


def gen_matrix(i, j, fill=0.):
    matrix = []
    for i in range(i):
        matrix.append([fill] * j)
    return np.array(matrix)


def gen_mask(i, j):
    matrix = []
    for i in range(i):
        matrix.append([True] * j)
    return np.array(matrix)


def gen_ematrix(n, inp=0):
    matrix = []
    for i in range(n):
        matrix.append([0 if k != i else 1 for k in range(n + inp)])
    return np.array(matrix)


def gen_mask_fft(n, m):
    arr = gen_mask(n, m)
    for i in range(n):
        for j in range(m):
            if (int(i >= n//2 ) - 0.5) * (int(j >= m//2 ) - 0.5) >= 0:
                arr[i, j] = 1
            else:
                arr[i, j] = 0
    return arr


def gen_func(name):
    if name == "tanh":
        return functions.tanh.func, functions.tanh.derv, functions.tanh.name
    elif name == "relu":
        return functions.relu.func, functions.relu.derv, functions.relu.name


def gen_id(n):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    res = ""
    for i in range(n):
        res += alphabet[random.randint(0, len(alphabet)-1)]
    return res


def merge_matrix(a, b, shape):
    new_nrow = shape[0]
    new_ncol = shape[1]
    res = gen_matrix(new_nrow, new_ncol)
    for r in range(new_nrow):
        for c in range(new_ncol):
            val = list()
            if r < a.shape[0] and c < a.shape[1]:
                val.append(a[r, c])
            if r < b.shape[0] and c < b.shape[1]:
                val.append(b[r, c])
            res[r, c] = sum(val)
    return res


def merge_matrix_mask(a, b, shape):
    new_nrow = shape[0]
    new_ncol = shape[1]
    res = gen_matrix(new_nrow, new_ncol)
    for r in range(new_nrow):
        for c in range(new_ncol):
            val = list()
            if r < a.shape[0] and c < a.shape[1]:
                val.append(a[r, c])
            if r < b.shape[0] and c < b.shape[1]:
                val.append(b[r, c])
            res[r, c] = np.prod(val)
    return res


def dropout_input(signal):
    drop = np.array([i < 1.5 and i > -1.5 for i in np.random.normal(0, 1, len(signal))]).astype(np.int16)
    return 1. / 0.87 * drop * signal


def dropout_output(weights, signal):
    drop = np.array([i < 1 and i > -1 for i in np.random.normal(0, 1, len(signal))]).astype(np.int16)
    return 1. / 0.667 * np.dot(weights * np.array([drop for i in range(weights.shape[0])]), signal)


def amp(inputs):
    return functions.f_tanh(functions.f_tanh(functions.f_tanh(inputs)))


class animator(object):
    def __init__(self):
        self.fig, (self.ax0) = plt.subplots(1, 1)
        self.ani = 0
        self.ax1 = self.ax0.twinx()
        self.line0, = self.ax0.plot(list(), list(), lw=0.5)
        self.line1, = self.ax1.plot(list(), list(), lw=0.5, color="red")
        # self.ax0.set_yscale('log')
        self.line = [self.line0, self.line1]
        self.xdata, self.ydata0, self.ydata1 = list(), list(), list()

    def run(self, data):
        # update the data
        epoch, y0, y1 = data
        # self.xdata, self.y0data, = [], []
        self.xdata.append(epoch)
        self.ydata0.append(y0)
        self.ydata1.append(y1)

        # self.ax0.set_ylim((10 ** np.ceil(max(-6, np.log10(min(self.ydata0))) - 2)), 10 ** np.ceil(np.log10(max(self.ydata0)) + 2))
        if min(self.ydata0) != np.nan and max(self.ydata0) != np.inf:
            self.ax0.set_ylim(min(self.ydata0), max(self.ydata0)*1.1)
        self.ax1.set_ylim(0, 1)
        self.ax0.set_xlim(-(max(self.xdata) + 0.1) * 0.1, (max(self.xdata) + 0.1) * 1.2)

        self.line[0].set_data(self.xdata, self.ydata0)
        self.line[1].set_data(self.xdata, self.ydata1)
        self.ax0.set_xlabel("epochs")
        self.ax1.set_ylabel("accuracy")
        self.ax0.set_ylabel("error")
        for ax in [self.ax0]:
            ax.figure.canvas.draw()
        return self.line

    def animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.run, self.arrange, interval=100, blit=True, repeat=False)
        plt.show()

    def arrange_for_animation(self, func):
        self.arrange = func

    def func_dammy(self):
        for i, j in enumerate(range(10000)):
            yield i, j, 0.5


class animator2(animator):
    def __init__(self):
        self.fig, (self.ax0) = plt.subplots(1, 1)
        self.line0 = self.ax0.scatter(list(), list(), lw=0.5)
        self.line = [self.line0]
        self.xdata, self.ydata0 = list(), list()

    def run(self, data):
        self.ax0.cla()
        # update the data
        data, ans = data
        self.ax0.text(-1, 0, "%s" % ans)

        xdata, ydata, size = [], [], []
        for cnt, d in enumerate(data[::-1]):
            for c, _d in enumerate(d):
                xdata.append(cnt)
                ydata.append(c)
                size.append(_d)
                self.ax0.text(cnt, c, "%6.4f" % _d)

        size = np.array(size)
        size = normalize(size) + 20

        # self.xdata.append(xdata)
        # self.ydata0.append(ydata)

        self.ax0.scatter(xdata, ydata, s=size, c=size, animated=False)

        self.ax0.set_xlabel("leyers")
        self.ax0.set_ylabel("nodes")
        self.ax0.set_xlim([-1, len(data)])
        self.ax0.set_title("aaa")
        self.ax0.figure.canvas.draw()
        return [self.ax0, ]

    def animation(self):
        self.ani = animation.FuncAnimation(self.fig, self.run, self.arrange, interval=100, blit=True, repeat=False)
        plt.show()

    def arrange_for_animation(self, func):
        self.arrange = func

    def func_dammy(self):
        for i, j in enumerate(range(10000)):
            yield i, j, 0.5

def get_pickle(name, num=3, reverse=1):
    """reverse=1: high -> low"""
    res = list()
    for f in os.listdir("./pickle"):
        if name in f:
            try:
                pnt = float(".".join(re.split("score|p|_", f)[-5:-3]))
            except:
                pnt = 0.
            res.append([f, pnt])
    fs = list(zip(*sorted(res, key=lambda x: x[1], reverse=reverse)))[0][:num]
    return ["./pickle/%s" % f for f in fs]


def get_pre_trained_model(name, num, reverse=0):
    units = list()
    ids = list()
    for f in get_pickle(name, num=100, reverse=reverse):
        with open(f, mode="rb") as ob:
            identity = ob.name.split("_")[2]
            print(ob, identity)
            ob = pickle.load(ob)
            if identity not in ids:
                units.append(ob)
                ids.append(identity)
            else:
                print("skipped %s" % f)
            if len(ids) >= num: break
    return units


def normalize(arr):
    arr = (arr - arr.mean()) / arr.std()
    # arr = (arr - min(arr)) / (max(arr) - min(arr))
    return arr

if __name__ == '__main__':
    pass
