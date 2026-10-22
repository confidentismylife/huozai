"""Finite-library arithmetic and deterministic selection for offline audit only."""
import numpy as np

BASE=12


def scores(risks,supported):
    r=np.asarray(risks,float);support=np.asarray(supported)
    if r.ndim!=4 or r.shape[:2]!=(2,17) or min(r.shape[2:])<1 or not np.isfinite(r).all() or support.shape!=(2,) or support.dtype!=bool or not support.any():
        raise ValueError("finite two-world17-plan risks with support required")
    means=r[support,:,:,0].mean(axis=2)
    worst=means.max(axis=0);robust=worst[BASE]-worst
    paired=(means[:,BASE,None]-means).min(axis=0)
    robust[BASE]=paired[BASE]=0
    return robust,paired


def diagnostic_source_scores(risks,true_source):
    r=np.asarray(risks,float)
    if r.ndim!=4 or r.shape[:2]!=(2,17) or true_source not in (0,1) or not np.isfinite(r).all():raise ValueError("finite risks and explicit diagnostic source required")
    means=r[true_source,:,:,0].mean(axis=1);g=means[BASE]-means;g[BASE]=0
    return g


def choose(gains,margin=0.):
    g=np.asarray(gains,float)
    if g.shape!=(17,) or not np.isfinite(g).all() or g[BASE]!=0 or not np.isfinite(margin) or margin<0:raise ValueError("finite17 relative scores and nonnegative margin required")
    s=g-margin;s[BASE]=0.;best=float(s.max());tied=np.flatnonzero(best-s<=1e-12)
    p=min(map(int,tied),key=lambda j:(j!=BASE,j))
    return p if s[p]>.001 else BASE


def actual_gains(outcomes):
    o=np.asarray(outcomes,float)
    if o.ndim!=3 or o.shape[0]!=17 or min(o.shape[1:])<1 or not np.isfinite(o).all():raise ValueError("finite17-plan actual outcomes required")
    loss=o[:,:,0].mean(axis=1);g=loss[BASE]-loss;g[BASE]=0
    return g
