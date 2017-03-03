import numpy as np
import random
from matplotlib import pylab as plt
import matplotlib.animation as animation


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


def gen_ematrix(n, inp=0):
    matrix = []
    for i in range(n):
        matrix.append([0 if k != i else 1 for k in range(n + inp)])
    return np.array(matrix)


class animator(object):
    def __init__(self):
        self.fig, (self.ax0) = plt.subplots(1, 1)
        self.ani = 0
        self.line0, = self.ax0.plot(list(), list(), lw=0.5)
        self.ax0.set_yscale('log')
        self.line = [self.line0]
        self.xdata, self.ydata = list(), list()

    def run(self, data):
        # update the data
        epoch, error = data
        # self.xdata, self.y0data, = [], []
        self.xdata.append(epoch)
        self.ydata.append(error)

        self.ax0.set_ylim((10 ** np.floor(np.log10(min(self.ydata))) - 1), 10 ** np.ceil(np.log10(max(self.ydata)) + 1))
        self.ax0.set_xlim(-(max(self.xdata) + 0.1) * 0.1, (max(self.xdata) + 0.1) * 1.2)

        self.line[0].set_data(self.xdata, self.ydata)
        self.ax0.set_title("%-.5f" % error)
        self.ax0.set_xlabel("epochs")
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
            yield i, j

if __name__ == '__main__':
    ani = animator()
    ani.arrange_for_animation(ani.func_dammy())
    ani.animation()
