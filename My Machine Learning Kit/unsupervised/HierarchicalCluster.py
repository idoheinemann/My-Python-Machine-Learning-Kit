import numpy as np

def copy(x):
    if isinstance(x,np.ndarray):
        return x
    if isinstance(x, int) or isinstance(x, float):
        return x
    cp = []
    for i in x:
        cp.append(copy(i))
    return cp

def hierarchical_cluster(inp, beta=0.0, min_c=None, max_c=None):
    """
    clusters the data with hierarchical clustering
    :param inp: data to cluster
    :param beta: a parameter indicating the significance of the size of two joined clusters. a larger beta would mean
    that clusters with a large size are not likely to be joined with other clusters with great size
    default to 0: cluster size is insignificant
    :param min_c: minimum amount of clusters
    :param max_c: maximum amount of clusters
    :return: an array of integers, each indicating the cluster of the input parameter with whom they share an index
    """
    prev_score = 0
    clusters = [[i] for i in inp]
    steps = [{"clusters": clusters}]

    if min_c is None:
        min_c = 1

    def min_distance(x, y):
        m = np.Inf
        for i in x:
            for j in y:
                d = np.sum((i-j)**2)
                if d < m:
                    m = d
        return np.sqrt(m)

    def stamp_mark(x, y):
        return min_distance(x, y) + (beta*min(len(x), len(y)))

    while len(clusters) > min_c:
        clusters = copy(clusters)
        join1, join2 = 0, 0
        min_score = np.Inf
        for i in xrange(len(clusters)):
            for j in xrange(i+1, len(clusters)):
                score = stamp_mark(clusters[i], clusters[j])
                if score < min_score:
                    min_score = score
                    join1 = i
                    join2 = j
        clusters[join1] += clusters[join2]
        del clusters[join2]
        steps.append({"clusters": clusters})
        steps[-2]['stamp_mark'] = min_score-prev_score
        prev_score = min_score

    steps = steps[:-1]

    if max_c is not None:
        steps = steps[max_c:]

    max_score = 0
    step = steps[0]['clusters']
    for i in steps:
        if i['stamp_mark'] > max_score:
            max_score = i['stamp_mark']
            step = i['clusters']

    def cluster_of(x):
        for i in xrange(len(step)):
            for j in step[i]:
                if (j == x).all():
                    return i

    return [cluster_of(x) for x in inp]

def main():
    data = np.array([[136,345],[109,362],[139,373],[129,70],[105,84],[121,84],[394,246],[386,257],[400,256]])
    clusters = hierarchical_cluster(data)
    print clusters

if __name__ == '__main__':
    main()