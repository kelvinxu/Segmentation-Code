"""
Implementation of Cardinality potentials.

Yujia Li, 10/2013
"""

import numpy as np

def cardinality_map_binary(unary, card_fn):
    """MAP inference / energy minimization for energy functions of the form:
            sum_i f_i(y_i) + f_c(sum_i y_i)         (*)
    where f_i are unary potentials, f_c is a cardinality potential and y_i's
    are binary variables.

    unary: N*2 matrix, unary[i,k] = f_i(k)
    card_fn: an arbitrary real value function that accepts either a single 
        number or a numpy matrix as input, and apply this function in an
        element-wise way.

    Return y that minimizes the energy function (*).  y is a N-element 1-d
        numpy integer array.
    """
    N, K = unary.shape
    assert K == 2

    fc = card_fn(np.arange(1,N+1))  # number of 1's from 1 to N
    f = unary[:,1] - unary[:,0]

    idx = f.argsort()
    f_sorted = f[idx].cumsum()

    f_sum = f_sorted + fc
    k = f_sum.argmin()
    if f_sum[k] > card_fn(0):   # compare to the all-zero case
        k = -1

    y = np.zeros(N, dtype=np.int)
    y[idx[:k+1]] = 1

    return y


def _test_card_fn(x):
    return np.abs(x - 10000)

def _test_cardinality_map_binary():
    unary = np.array([
        [3, 0],
        [2, 2],
        [3, 1],
        [1, 3],
        [3, 2],
        [3, 4],
        [3, 0]
        ])
    y = cardinality_map_binary(unary, _test_card_fn)

    print y
    print 'Energy: %g' % (_test_card_fn(y.sum()) + unary[np.arange(unary.shape[0]),y].sum())

    import time

    n_vars = 100000
    unary = np.random.randn(n_vars, 2)
    t_start = time.time()
    y = cardinality_map_binary(unary, _test_card_fn)
    print 'Running MAP inference for %d variables... time %.2f' % (n_vars, time.time() - t_start)

if __name__ == '__main__':
    _test_cardinality_map_binary()
