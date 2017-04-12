import numpy
import scipy.stats
from PIL import Image


# class myfilter(object):
#     def __init__(self, kind, sigma, **kwargs):
#         x = numpy.arange(0, 100)
#         y = numpy.arange(0, 100)
#         xx, yy = numpy.meshgrid(x, y)

#         if kind == "gausiun":
#             arr = self.gaussian(xx, yy, sigma=sigma)
#             self.kernel = arr / arr.max()
#             # return arr

#         elif kind == "deg":
#             arr = self.deg(xx, yy, sigma=sigma, deg=kwargs["deg"])
#             self.kernel = arr / arr.max()
#             # return arr

def gaussian(sigma=0.2):
    x = numpy.arange(0, 100)
    y = numpy.arange(0, 100)
    x, y = numpy.meshgrid(x, y)
    x = numpy.array(x).flatten()
    y = numpy.array(y).flatten()
    x = x / 100. - 0.5
    y = y / 100. - 0.5
    arr = scipy.stats.norm.pdf(0, loc=numpy.sqrt(x**2 + y**2), scale=sigma).reshape(100, 100)
    # Image.fromarray(arr * 255).show()
    # print(arr)
    return Image.fromarray(arr)

def deg(sigma=0.1, deg=45):
    deg = - deg / 180. * numpy.pi
    x = numpy.arange(0, 100)
    y = numpy.arange(0, 100)
    x, y = numpy.meshgrid(x, y)
    x = numpy.array(x).flatten()
    y = numpy.array(y).flatten()
    x = x / 100. - 0.5
    y = y / 100. - 0.5
    x, y = numpy.dot(numpy.array([x, y]).T, numpy.array([[numpy.cos(deg), -numpy.sin(deg)], [numpy.sin(deg), numpy.cos(deg)]])).T
    arr = scipy.stats.norm.pdf(0, loc=abs(y), scale=sigma).reshape(100, 100)
    Image.fromarray(arr * 255).show()
    print(arr)

    return Image.fromarray(arr)



if __name__ == '__main__':
    deg()
