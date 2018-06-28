import numpy as np


class GuassianKernel:
    def __init__(self, points, sigma=1):
        self.points = np.copy(points)
        self.sigma = sigma

    def __call__(self, p1, p2):
        p = p1-p2
        return np.exp(-np.sum(p*p)/self.sigma)

    def prepare(self, points):
        values = []
        for i in points:
            tmp = []
            for j in self.points:
                tmp.append(self(i, j))
            values.append(tmp)
        return np.array(values)
