import numpy as np


def sigmoid(x, d=0):
    return x*(x-1) if d else 1/(1+np.exp(-x))


def tanh(x, d=0):
    return 1-x*x if d else np.tanh(x)


def relu(x, d=0):
    e = np.exp(x)
    return (e-1)/e if d else np.log(e+1)


def linear_relu(x, d=0):
    return 1 if d else x if x > 0 else 0


def create_parametric_relu(a):
    def parametric_relu(x, d=0):
        return 1 if d else x if x > 0 else a*x
    return parametric_relu


def linear(x, d=0):
    return 1 if d else x


def arctan(x, d=0):
    return 1/(1+np.tan(x)**2) if d else np.arctan(x)


def softmax(x, d=0):
    if d:
        return x*(1-x)
    e = np.exp(x)
    return e/(1+e)

