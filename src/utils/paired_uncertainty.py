"""Paired circular moving-block bootstrap for a regularly sampled loss difference."""
import numpy as np


def paired_block_interval(differences, block_hours, repetitions=2000, seed=1701):
    d=np.asarray(differences,dtype=float)
    if d.ndim!=1 or not np.isfinite(d).all() or not len(d):raise ValueError('Finite paired differences required')
    n=len(d);b=int(block_hours)
    if b<1 or b>n or repetitions<2:raise ValueError('Invalid bootstrap settings')
    rng=np.random.default_rng(seed)
    full,remainder=divmod(n,b)
    extended=np.r_[d,d[:b]];prefix=np.r_[0.,np.cumsum(extended)]
    starts=rng.integers(0,n,size=(repetitions,full+bool(remainder)))
    samples=(prefix[starts[:,:full]+b]-prefix[starts[:,:full]]).sum(axis=1)
    if remainder:samples+=prefix[starts[:,-1]+remainder]-prefix[starts[:,-1]]
    samples/=n
    lower,upper=np.quantile(samples,[.025,.975])
    return dict(difference=float(d.mean()),lower_95=float(lower),upper_95=float(upper),
        block_hours=b,repetitions=repetitions,count=n)
