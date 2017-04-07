import numpy
import scipy.stats
from PIL import Image


class filter(object):
    def __init__(self, kind, sigma, **kwargs):
        x = numpy.arange(0, 100)
        y = numpy.arange(0, 100)
        xx, yy = numpy.meshgrid(x, y)

        if kind == "gausiun":
            arr = self.gaussian(xx, yy, sigma=sigma)
            arr = arr / arr.max()
            # return arr

        elif kind == "deg":
            arr = self.deg(xx, yy, sigma=sigma, deg=kwargs["deg"])
            arr = arr / arr.max()
            Image.fromarray(arr * 255).show()
            # return arr

    def gaussian(self, x, y, sigma=0.3):
        x = numpy.array(x).flatten()
        y = numpy.array(y).flatten()
        x = x / 100. - 0.5
        y = y / 100. - 0.5
        return scipy.stats.norm.pdf(0, loc=numpy.sqrt(x**2 + y**2), scale=sigma)

    def deg(self, x, y, sigma=0.01, deg=0):
        deg = - deg / 180. * numpy.pi
        x = numpy.array(x).flatten()
        y = numpy.array(y).flatten()
        x = x / 100. - 0.5
        y = y / 100. - 0.5
        y = numpy.maximum(-0.4, numpy.minimum(y, 0.4))
        x, y = numpy.dot(numpy.array([x, y]).T, numpy.array([[numpy.cos(deg), -numpy.sin(deg)], [numpy.sin(deg), numpy.cos(deg)]])).T
        return scipy.stats.norm.pdf(0, loc=numpy.sqrt(y**2), scale=sigma).reshape(100, 100)



if __name__ == '__main__':
    filter("deg", 0.1, deg=10)
