# fit sbms

import graph_tool.all as gt
import numpy as np


def refine_minimize_nested_blockmodel_dl(g, iters=1000, sweeps=10):
    state = gt.minimize_nested_blockmodel_dl(g)
    for i in range(iters):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=sweeps)
    return state


def refine_minimize_blockmodel_dl(g, dc, pp, nested, refine=True, iters=1000, sweeps=10):
    if nested:
        if pp:
            raise ValueError('Nested version of pp variant not implemented.')
        state = gt.minimize_nested_blockmodel_dl(g, state_args=dict(deg_corr=dc))
    else:
        if pp:
            if not dc:
                raise ValueError('Non-degree-corrected version of pp variant not implemented.')
            state = gt.minimize_blockmodel_dl(g, state=gt.PPBlockState)
        else:
            state = gt.minimize_blockmodel_dl(g, state_args=dict(deg_corr=dc))
    if refine:
        for i in range(iters):
            state.multiflip_mcmc_sweep(beta=np.inf, niter=sweeps)
    return state


def cluster_partitions(bs, nested):
    # Infer partition modes
    pmode = gt.ModeClusterState(bs, nested=nested)

    # Minimize the mode state itself
    gt.mcmc_equilibrate(pmode, wait=1, mcmc_args=dict(niter=1, beta=np.inf))
    return pmode


def sample_partitions(g, nested, deg_corr,
                      n_partitions,
                      wait=1000,
                      force_niter=1000,
                      include_dls=True,
                      multigraph_dl=True):
    global bs, dls
    if nested:
        state = gt.NestedBlockState(g, state_args=dict(deg_corr=deg_corr))
    else:
        state = gt.BlockState(g, deg_corr=deg_corr)

    # Equilibration
    gt.mcmc_equilibrate(state, wait=wait, force_niter=force_niter, mcmc_args=dict(niter=10))

    bs = []
    dls = []

    def collect_partitions(s):
        global bs, dls
        if nested:
            bs.append(s.get_bs())
        else:
            bs.append(s.get_state().a.copy())
        if include_dls:
            dls.append(s.entropy(multigraph=multigraph_dl))

    gt.mcmc_equilibrate(state,
                        force_niter=n_partitions,
                        mcmc_args=dict(niter=10),
                        callback=collect_partitions)

    if include_dls:
        return state, bs, dls
    else:
        return state, bs


