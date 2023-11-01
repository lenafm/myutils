import graph_tool.all as gt
import numpy as np
import scipy.special as sp

from math import lgamma, e
from collections import defaultdict

# The description length calculations in this file are based on the work by Tiago Peixoto, as described in
# Peixoto, Tiago P. "Entropy of stochastic blockmodel ensembles." Physical Review E 85.5 (2012): 056122
# and implemented in the graph_tool library.


def get_block_graph(g, b):
    B = len(set(b))
    cg, br, vc, ec, av, ae = gt.condensation_graph(g, b,
                                                   self_loops=True)
    cg.vp.count = vc
    cg.ep.count = ec
    rs = np.setdiff1d(np.arange(B, dtype="int"), br.fa,
                      assume_unique=True)
    if len(rs) > 0:
        cg.add_vertex(len(rs))
        br.fa[-len(rs):] = rs

    cg = gt.Graph(cg, vorder=br)
    return cg


def get_block_stats(state=None, g=None, b=None, verbose=True):
    if state is None:
        if verbose:
            if g is None or b is None:
                raise Exception("Either state or both graph g and partition b must be specified.")
        if not isinstance(b, gt.VertexPropertyMap):
            if np.max(b) > len(set(b)) - 1:
                b = gt.contiguous_map(np.array(b))
            b_new = g.new_vp('int')
            b_new.a = np.array(b)
            b = b_new.copy()
        else:
            b = gt.contiguous_map(b)
        bg = get_block_graph(g, b)
        B = bg.num_vertices()
        N = g.num_vertices()
        E = g.num_edges()
        mrs = bg.ep["count"]
        wr = bg.vp["count"]
        mrp = bg.degree_property_map("out", weight=mrs)
        mrm = mrp
        ers = gt.adjacency(bg, mrs)
    else:
        if verbose:
            if g is not None or b is not None:
                print('Graph g and or partition b was specified although state was specified - state is being used.')
        b = gt.contiguous_map(state.get_blocks())
        state = state.copy(b=b)
        bg = state.get_bg()
        B = state.get_B()
        N = state.get_N()
        mrs = state.mrs
        wr = state.wr
        mrp = state.mrp
        mrm = state.mrm
        E = sum(mrp.a) / 2
        ers = state.get_matrix().todense()
    return bg, b, N, E, B, mrs, wr, mrp, ers


def calculate_dl(state=None, g=None, b=None, dc=True, exact=True,
                 blockstate='BlockState',
                 uniform=False, degree_dl_kind='distributed',
                 verbose=False):
    H_ensemble = sparse_entropy(state=state, g=g, b=b, dc=dc, exact=exact)
    H_partition = partition_dl(state=state, g=g, b=b)
    H_edges = edges_dl(state=state, g=g, b=b, blockstate=blockstate, uniform=uniform)
    H_degrees = degree_dl(state=state, g=g, b=b, degree_dl_kind=degree_dl_kind)
    H = H_ensemble + H_partition + H_edges
    txt1 = 'H_ensemble + H_partition + H_edges'
    txt2 = '{} + {} + {}'.format(H_ensemble, H_partition, H_edges)
    if dc:
        H += H_degrees
        txt1 = '{} + {}'.format(txt1, 'H_degrees')
        txt2 = '{} + {}'.format(txt2, H_degrees)
    if verbose:
        print(txt1)
        print(txt2)
        print('total dl: {}'.format(H))
        print('--------')
    return H


def partition_dl(state=None, g=None, b=None):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g, b)
    nrs = wr.a
    S = 0
    S += lbinom(N - 1, B - 1)  # log(binom(N-1, B-1))
    S += lgamma(N + 1)  # log(N!)
    for nr in nrs:
        S -= lgamma(nr + 1)  # -sum_r (log(nr!))
    S += safelog(N)
    return S


def degree_dl(state=None, g=None, b=None, degree_dl_kind='uniform'):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state=state, g=g, b=b)
    S = 0
    if degree_dl_kind == 'uniform':
        for v in bg.vertices():
            S += lbinom(wr[v] + mrp[v] - 1, mrp[v])
    elif degree_dl_kind == 'distributed':
        deg_hists = get_degree_histograms(state=state, g=g, b=b)
        for r in bg.vertices():
            hist = deg_hists[int(r)]
            for k in hist:
                S -= lgamma(k[1] + 1)
            q = num_partitions(mrp[r], wr[r])
            S += safelog(float(q))
            S += lgamma(wr[r] + 1)
    else:
        raise Exception("Degree dl kind must be one of 'uniform' or 'distributed'.")
    return S


def degree_dl_distributed(state=None, g=None, b=None):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state=state, g=g, b=b)
    S = 0
    deg_hists = get_degree_histograms(state=state, g=g, b=b)
    for r in bg.vertices():
        hist = deg_hists[int(r)]
        for k in hist:
            S -= lgamma(k[1] + 1)
        S += safelog(num_partitions(mrp[r], wr[r]))
        S += lgamma(wr[r] + 1)
    return S


# the uniform version (used in the PP paper)
def degree_dl_uniform(state=None, g=None, b=None):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g, b)
    S = 0
    for v in bg.vertices():
        S += lbinom(wr[v] + mrp[v] - 1, mrp[v])
    return S


# this corresponds to the case where the prior on the partition simply is B^-N
def partition_dl_simple(state=None, g=None, b=None):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g, b)
    return N * safelog(B)


def edges_dl(state=None, g=None, b=None, blockstate='BlockState', uniform=True):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g, b)
    if blockstate == 'BlockState':
        NB = (B * (B + 1)) / 2
        S = lbinom(NB + E - 1, E)
    elif blockstate == 'PPBlockState':
        diag = np.diag_indices(ers.shape[0])
        sum_diag = np.sum(ers[diag])
        sum_off_diag = np.sum(ers) - sum_diag
        e_in = sum_diag / 2
        e_out = sum_off_diag / 2
        S = 0
        if uniform:
            S -= lgamma(e_in + 1)
            S -= lgamma(e_out + 1)
            S += e_in * safelog(B)
            S += e_out * lbinom(B, 2)
            if B > 1:
                S += safelog(E + 1)
            for e in bg.edges():
                r = e.source()
                s = e.target()
                S += lgamma(mrs[e] + 1)  # not sure if this should if r >= s or going through all
        else:
            S -= lgamma(e_out + 1)
            S += e_out * lbinom(B, 2)
            if B > 1:
                S += safelog(E + 1)
            S += lbinom(B + e_in - 1, e_in)
            for e in bg.edges():
                r = e.source()
                s = e.target()
                if r != s:  # not sure if this should be != or <
                    S += lgamma(mrs[e] + 1)
    else:
        raise Exception('Blockstate needs to be one of BlockState or PPBlockState.')
    return S


# def edges_dl_pp(state=None, g=None, b=None, uniform=True):
#     bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g, b)
#     diag = np.diag_indices(ers.shape[0])
#     sum_diag = np.sum(ers[diag])
#     sum_off_diag = np.sum(ers) - sum_diag
#     e_in = sum_diag/2
#     e_out = sum_off_diag/2
#     S = 0
#     if uniform:
#         S -= lgamma(e_in+1)
#         S -= lgamma(e_out+1)
#         S += e_in * safelog(B)
#         S += e_out * lbinom(B, 2)
#         if B > 1:
#             S += safelog(E+1)
#         for e in bg.edges():
#             r = e.source()
#             s = e.target()
#             S += lgamma(mrs[e]+1) # not sure if this should if r >= s or going through all
#     else:
#         S -= lgamma(e_out+1)
#         S += e_out * lbinom(B, 2)
#         if B > 1:
#             S += safelog(E+1)
#         S += lbinom(B + e_in - 1, e_in)
#         for e in bg.edges():
#             r = e.source()
#             s = e.target()
#             if r != s: # not sure if this should be != or <
#                 S += lgamma(mrs[e]+1)
#     return S


def sparse_entropy(state=None, g=None, b=None, dc=False, exact=True):
    if state is not None:
        bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state=state, g=None, b=b)
    else:
        bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state=state, g=g, b=b)
    S = 0
    if exact:
        for e in bg.edges():
            r = e.source()
            s = e.target()
            S += eterm_exact(r, s, mrs[e])
        for v in bg.vertices():
            S += vterm_exact(mrp[v], wr[v], dc)
    else:
        for e in bg.edges():
            r = e.source()
            s = e.target()
            S += eterm(r, s, mrs[e])
        for v in bg.vertices():
            S += vterm(mrp[v], mrp[v], wr[v])
    if not exact:
        S += E
    if dc:
        for v in g.vertices():
            S += get_degree_entropy(v)
    return S


def sparse_entropy_exact(state=None, g=None, b=None, dc=False):
    g_cp = g.copy()
    if state is not None:
        g_cp = None
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g_cp, b)
    S = 0
    for e in bg.edges():
        r = e.source()
        s = e.target()
        S += eterm_exact(r, s, mrs[e])
    for v in bg.vertices():
        S += vterm_exact(mrp[v], mrp[v], wr[v], dc)
    if dc:
        for v in g.vertices():
            S += get_degree_entropy(v)
    return S


def sparse_entropy_approx(state=None, g=None, b=None, dc=False):
    g_cp = g.copy()
    if state is not None:
        g_cp = None
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g_cp, b)
    S = 0
    for e in bg.edges():
        r = e.source()
        s = e.target()
        S += eterm(r, s, mrs[e])
    for v in bg.vertices():
        S += vterm(mrp[v], mrp[v], wr[v], dc)
    if dc:
        for v in g.vertices():
            S += get_degree_entropy(v)
    return S + E


def eterm_exact(r, s, mrs):
    val = lgamma(mrs + 1)
    if r != s:
        return -val
    else:
        return -val - (mrs * safelog(2))


def vterm_exact(mrp, wr, dc=False):
    if dc:
        return lgamma(mrp + 1)
    else:
        return mrp * safelog(wr)


def get_degree_entropy(v):
    k = v.out_degree()
    return -lgamma(k + 1)


def eterm(r, s, mrs):
    if r == s:
        mrs *= 2
    val = xlogx(mrs)
    if r != s:
        return -val
    else:
        return -val / 2


def vterm(mrp, mrm, wr, dc=False):
    one = 0.5
    if dc:
        return one * (xlogx(mrm) + xlogx(mrp))
    else:
        return one * (mrm * safelog(wr) + mrp * safelog(wr))


def dense_entropy(state=None, g=None, b=None):
    bg, b, N, E, B, mrs, wr, mrp, ers = get_block_stats(state, g, b)
    S = 0
    for e in bg.edges():
        r = e.source()
        s = e.target()
        term = eterm_dense(r, s, mrs[e], wr[r], wr[s])
        S += term
    return S


def eterm_dense(r, s, mrs, wr_r, wr_s):
    if r != s:
        nrns = wr_r * wr_s
    else:
        nrns = (wr_r * (wr_r - 1)) / 2
    return lbinom(nrns, mrs)


def num_partitions(n, k):
    """
    Returns the number of partitions of integer n into at most k parts.
    """
    # Initialize table with base cases
    table = [[1] * (n + 1) for _ in range(k + 1)]

    # Fill in table using dynamic programming
    for i in range(2, k + 1):
        for j in range(1, n + 1):
            if j >= i:
                table[i][j] = table[i - 1][j] + table[i][j - i]
            else:
                table[i][j] = table[i - 1][j]

    return table[k][n]


def get_degree_histograms(state=None, g=None, b=None):
    if g is None:
        raise Exception("Need to specify graph g.")
    if state is None and b is None:
        raise Exception("Need to specify either state or partition b.")
    if b is None:
        b = state.b
    hist = defaultdict(dict)
    degrees = g.get_out_degrees(g.get_vertices())
    for v in g.vertices():
        r = b[v]
        if not r in hist: hist[r] = {}
        deg = degrees[int(v)]
        if not deg in hist[r]: hist[r][deg] = 0
        hist[r][deg] += 1
    hist = {k: [(key, val) for key, val in v.items()] for k, v in hist.items()}
    return hist


def lbinom(n, k):
    return safelog(sp.binom(n, k))


def safelog(x, base=e):
    if x == 0:
        return 0
    if base == 2:
        return np.log2(x)
    elif base == e:
        return np.log(x)


def xlogx(x, base=e):
    return x * safelog(x, base=base)