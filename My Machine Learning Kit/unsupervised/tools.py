import numpy as np


def centers(x, c):
    cents = [np.zeros_like(x[0]) for _ in xrange(np.amax(c))]
    _, counts = np.unique(c, return_counts=True)
    for i in xrange(len(c)):
        cents[c[i]] += x[i]
    for i in xrange(len(counts)):
        cents[i] /= counts[i]
    return cents


def split_by_clusters(x, c):
    a = [[]]*np.amax(c)
    for i in xrange(len(x)):
        a[c[i]].append(x[i])
    return a
