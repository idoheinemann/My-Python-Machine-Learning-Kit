import numpy as np


class BinaryTree:
    def __init__(self, value=None, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

    def __len__(self):
        return 1 + (len(self.right) if self.right is not None else 0) + (len(self.left) if self.left is not None else 0)


def is_numeric(x):
    return isinstance(x, int) or isinstance(x, float)


def get_keys(table):
    return set(map(lambda x: x[-1], table))


def counts(table):
    keyw = get_keys(table)
    counters = {}
    for i in keyw:
        counters[i] = 0
    for i in table:
        counters[i[-1]] += 1
    return counters


def get_values(table, row):
    return set(map(lambda x: x[row], table))


def gini_index(table):
    gini = 1
    lng = float(len(table))
    for i in counts(table).values():
        gini -= (i/lng)**2
    return gini


def find_best_split(table):
    mini_gini = gini_index(table)
    question = Leaf(table)
    for i in xrange(len(table[0])-1):
        v = list(get_values(table, i))
        for j in xrange(len(v)):
            q = Question(value=v[j], index=i)
            true_table, false_table = q.split(table)
            tmp_gini = (gini_index(true_table)*len(true_table)+gini_index(false_table)*len(false_table))/len(table)
            if tmp_gini < mini_gini:
                question = q
                mini_gini = tmp_gini
    return question


def add_question(node, table, downs_left=None):
    if downs_left == 0:
        node.value = Leaf(table)
    else:
        node.value = find_best_split(table)
        if isinstance(node.value, Question):
            node.left = BinaryTree()
            node.right = BinaryTree()
            true_table, false_table = node.value.split(table)
            add_question(node.left, true_table, downs_left-1 if downs_left is not None else None)
            add_question(node.right, false_table, downs_left - 1 if downs_left is not None else None)


class Question:
    def __init__(self, value, index, numeric=None):
        self.value = value
        self.index = index
        self.numeric = is_numeric(value) if numeric is None else numeric

    def __call__(self, row):
        if self.numeric:
            return row[self.index] >= self.value
        return row[self.index] == self.value

    def split(self, table):
        true_table = []
        false_table = []
        for i in table:
            true_table.append(i) if self(i) else false_table.append(i)
        return true_table, false_table


class Leaf:
    def __init__(self, table):
        self.odds = counts(table)
        for i in self.odds.keys():
            self.odds[i] /= float(len(table))

    def __call__(self):
        return self.odds


class DecisionTree:
    def __init__(self):
        self.root = BinaryTree()

    def __call__(self, value):
        node = self.root
        while isinstance(node.value, Question):
            node = node.left if node.value(value) else node.right
        return node.value()

    def train(self, table, max_layers=None):
        add_question(self.root, table, max_layers)

    def hypot(self, table):
        a = []
        for i in table:
            a.append(self(i))
        return a


def main():
    training_data = [
        ['Green', 3, 'Apple'],
        ['Yellow', 2, 'Apple'],
        ['Red', 1, 'Grape'],
        ['Red', 1, 'Grape'],
        ['Yellow', 3, 'Lemon'],
    ]
    tree = DecisionTree()
    tree.train(training_data)
    for i in training_data:
        print tree(i)


if __name__ == '__main__':
    main()