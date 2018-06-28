import numpy as np


def k_cluster(inp, k, max_iter=100, validate=True):
    clusters = [np.random.randint(k) for _ in inp]

    def distance(x, y):
        return np.linalg.norm(x-y)

    def closest(x):
        ind = 0
        dist = distance(x, centers[0])
        for i in xrange(1, len(centers)):
            d = distance(x, centers)
            if d < dist:
                dist = d
                ind = i
        return ind

    if validate:
        from copy import copy
        for _ in xrange(max_iter):
            prev = copy(clusters)
            centers = [np.zeros_like(inp[0]) for _ in xrange(k)]
            counts = list((0,)*k)
            for i in xrange(len(inp)):
                counts[i] += 1
                centers[i] += inp[i]
            for i in xrange(k):
                centers[i] /= float(counts[i])
            for i in xrange(len(inp)):
                clusters[i] = closest(inp[i])
            if (prev == clusters).all():
                return clusters
    else:
        for _ in xrange(max_iter):
            centers = [np.zeros_like(inp[0]) for _ in xrange(k)]
            counts = list((0,)*k)
            for i in xrange(len(inp)):
                counts[i] += 1
                centers[i] += inp[i]
            for i in xrange(k):
                centers[i] /= float(counts[i])
            for i in xrange(len(inp)):
                clusters[i] = closest(inp[i])

    return clusters

def main():
    data = np.array([[136,345],[109,362],[139,373],[129,70],[105,84],[121,84],[394,246],[386,257],[400,256]])
    clusters = k_cluster(data,3)
    print clusters

if __name__ == '__main__':
    main()
