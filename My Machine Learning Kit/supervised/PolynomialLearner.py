import numpy as np
from NonLinearityFunctions import linear


def calc_theta_num(n, k):
    return np.math.factorial(n+k)/(np.math.factorial(k)*np.math.factorial(n))


def inv_theta_num(n, x):
    k = 0
    while calc_theta_num(n, k) < x:
        k += 1
    return k


class Polynomial:
    def __init__(self, dim, deg, func):
        self.degree = deg
        self.dimension = dim
        self.function = func
        self.theta = [np.random.rand() for _ in xrange(calc_theta_num(dim, deg))]

    def __call__(self, point):
        value = 0
        deg = 0
        degrees_arr = [0] * self.dimension

        for i in xrange(len(self.theta)):
            theta_mul = self.theta[i]
            for n in xrange(len(degrees_arr)):
                theta_mul *= point[n] ** degrees_arr[n]
            value += theta_mul
            if degrees_arr[-1] == deg:
                deg += 1
                degrees_arr = [deg] + ([0] * (self.dimension - 1))
            else:
                for j in xrange(len(degrees_arr) - 2, -1, -1):
                    if degrees_arr[j] != 0:
                        degrees_arr[j] -= 1
                        degrees_arr[j + 1] += 1
                        for t in xrange(j+2, len(degrees_arr)):
                            degrees_arr[j + 1] += degrees_arr[t]
                            degrees_arr[t] = 0
                        break

        return self.function(value)

    def cost(self, x, y):
        j = 0
        for i in xrange(len(x)):
            j += (self(x)-y)**2
        return j

    def hypot(self, points):
        a = []
        for i in points:
            a.append(self(i))
        return a

    def theta_x(self, point, index):
        if index == 0:
            return 1
        deg = inv_theta_num(self.dimension, index + 1) - 1  # degree
        degree_arr = [deg + 1]+([0]*(self.dimension-1))
        for i in xrange(calc_theta_num(self.dimension, deg), index):
            for j in xrange(len(degree_arr)-2,-1,-1):
                if degree_arr[j] != 0:
                    degree_arr[j] -= 1
                    degree_arr[j+1] += 1
                    for t in xrange(j+2, len(degree_arr)):
                        degree_arr[j+1] += degree_arr[t]
                        degree_arr[t] = 0
                    break
        thx = 1
        for k in xrange(len(degree_arr)):
            thx *= point[k] ** degree_arr[k]
        return thx

    def iteration(self, x, y, alpha=0.001):
        derivatives = [0]*len(self.theta)
        for i in xrange(len(x)):
            output = self(x[i])
            for j in xrange(len(self.theta)):
                derivatives[j] += (output-y[i])*self.function(output, 1)*self.theta_x(x[i],j)
        for i in xrange(len(self.theta)):
            self.theta[i] -= alpha*derivatives[i]


class PolynomialLearner:
    def __init__(self, out, dim=None, deg=None, kernel=None, functions=None):
        assert (dim is not None and deg is not None) or kernel is not None
        self.kernel = kernel
        self.dimension = len(kernel) if kernel else dim
        self.degree = deg if deg is not None else 1
        if hasattr(functions, '__iter__') or hasattr(functions, '__getitem__'):
            assert out == len(functions)
        else:
            functions = (functions if functions is not None else  linear,)*out
        self.polynomials = [Polynomial(dim=self.dimension, deg=self.degree, func=functions[i]) for i in xrange(out)]
        self.outputs = out

    def __len__(self):
        return len(self.polynomials)

    def __call__(self, point):
        out = np.zeros(self.outputs)
        if self.kernel is not None:
            point = self.kernel.prepare(point)
        for i in xrange(len(self.polynomials)):
            out[i] = self.polynomials[i](point)
        return out

    def hypot(self, points):
        a = []
        for i in self.polynomials:
            a.append(i.hypot(points if self.kernel is None else self.kernel.prepare(points)))
        return np.array(a).T

    def iteration(self, x, y, alpha=0.001):
        y = np.array(y).T
        for i in xrange(len(self.polynomials)):
            self.polynomials[i].iteration(x, y[i], alpha)