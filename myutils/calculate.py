import numpy as np


def safelog(x):
    if x == 0:
        return 0
    else:
        return np.log(x)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))
