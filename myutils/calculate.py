import numpy as np
from scipy.stats import entropy as H


def safelog(x):
    if x == 0:
        return 0
    else:
        return np.log(x)


def jaccard_similarity(list1, list2):
    s1 = set(list1)
    s2 = set(list2)
    return float(len(s1.intersection(s2)) / len(s1.union(s2)))


def gJSD(p, weights=None, verbose=False):
    js_left = np.zeros(len(p[0]))
    js_right = 0
    for i, pd in enumerate(p):
        if weights is None:
            weight = 1/len(p)
        else:
            weight = weights[i]
        js_left += pd * weight
        weighted_ent = weight * H(pd)
        js_right += weighted_ent
        if verbose:
            print('entropy of prob {}: {}, weighted by {}'.format(i, weighted_ent, weight))
    H_mixed = H(js_left)
    jsd = H_mixed - js_right
    if verbose:
        print('---------------------------')
        print('entropy mixed distribution: {}'.format(H_mixed))
        print('sum of individual entropies: {}'.format(js_right))
        print('---------------------------')
        print('absolute gJSD: {}'.format(jsd))
    return jsd


def pw_diff(ls):
    diff = []
    for i, el_i in enumerate(ls):
        for j, el_j in enumerate(ls):
            if j > i:
                diff.append(abs(el_i-el_j))
    return diff
