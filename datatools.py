"""A set of simple tools used to process data.

Implemented tools:
- PCA, principle component analysis

Yujia Li, 07/2013
"""

import numpy as np
import scipy.linalg as la

_SMALL_CONSTANT = 1e-8

def pca(x, K):
    """(x, K) --> (xnew, basis, xmean)

    x: N*D is the data matrix, each row is a data vector
    K: an integer, the dimensionality of the low dimensional space to project

    xnew: N*K projected data matrix
    basis: D*K matrix, each column is a basis vector for the low dimensional space
    xmean: 1-D vector, the mean vector of x
    """

    xmean = x.mean(axis=0)

    X = x - xmean
    [w, basis] = np.linalg.eigh(X.T.dot(X))
    idx = np.argsort(w)
    idx = idx[::-1]
    basis = basis[:,idx[:K]]

    xnew = X.dot(basis)

    return xnew, basis, xmean

def pca_dim_reduction(x, basis, xmean=None):
    """(x, basis, xmean) --> xnew
    Dimensionality reduction with PCA.

    x: N*D data matrix
    basis: D*K basis matrix
    xmean: 1-D vector, mean vector used in PCA, if not set, use the mean of x instead

    xnew: N*K new data matrix
    """

    if xmean == None:
        xmean = x.mean(axis=0)

    xnew = (x - xmean).dot(basis)
    return xnew

def zero_mean_normalize(x, xmean=None, std=None):
    """(x, xmean=None, std=None) --> xnew

    Subtract mean from x, then divide by standard deviation, so that each
    dimension is roughly normally distributed.

    x: N*D data matrix
    xmean: D-dimensional vector, mean of each dimension
    std: D-dimensional vector, std of each dimension
    xnew: normalized N*D data matrix
    """

    if xmean == None:
        xmean = x.mean(axis=0)
    if std == None:
        std = x.std(axis=0) + _SMALL_CONSTANT

    return (x - xmean) / std

def scale_normalization(x, std=None):
    """(x, std=None) --> xnew

    Devide each dimension of x by the standard deviation of that dimension so 
    that all dimensions have roughly the same scale.

    x: N*D data matrix
    std: D-dimensional vector. It will be used as standard deviation if given.
    xnew: normalized N*D data matrix
    """
    if std == None:
        std = x.std(axis=0) + _SMALL_CONSTANT

    return x / std

def list_to_mat(xlist):
    """(xlist) --> x

    Concatenate a list of matrices into a full matrix. All matrices in the list
    should have the same width, i.e. of size N*K where K is the same for all 
    matrices.

    The size of x will be (sum_i N_i) * K.
    """
    assert (len(xlist) > 0)

    N, K = xlist[0].shape
    for i in range(1, len(xlist)):
        N += xlist[i].shape[0]

    x = np.empty((N,K), xlist[0].dtype)
    i_start = 0
    for i in range(len(xlist)):
        x[i_start:i_start + xlist[i].shape[0]] = xlist[i]
        i_start += xlist[i].shape[0]

    return x

def list_to_vec(xlist):
    """(xlist) --> x

    Concatenate a list of vectors into a long vector. All vectors are 1-d 
    numpy ndarrays of the same type.
    """
    assert (len(xlist) > 0)
    
    total_len = sum([v.size for v in xlist])
    x = np.empty(total_len, dtype=xlist[0].dtype)

    i_start = 0
    for i in range(len(xlist)):
        x[i_start:i_start + xlist[i].size] = xlist[i]
        i_start += xlist[i].size

    return x

def list_mode(x, K):
    """Find the mode (mose frequent element) in x. x can be indexed by one 
    integer, i.e. a list or 1-d array. Each element in x is restricted to be
    in range 0 to K-1."""
    count = np.zeros(K, dtype=np.int)
    for i in xrange(len(x)):
        count[x[i]] += 1
    return count.argmax()

def switch_row_column_major(x, mat_size):
    """(x, mat_size) --> new_x
    Switch the data ordering of matrices between row major and column major.
    x: N*D matrix, each row stores a matrix in row major or column major order.
    mat_size: the original shape of the matrices stored in x
    new_x: same size as x, but with row major switched to column major or the
        other way around.
    """
    new_x = x.copy()

    for i in range(x.shape[0]):
        new_x[i] = x[i].reshape(mat_size).T.reshape([1, x.shape[1]])

    return new_x


class Preprocessor(object):
    """Base class for preprocessors."""
    def __init__(self, x=None, prev=None, **kwargs):
        """Construct preprocessor.  Can use some data x, or chained with
        another preprocessor."""
        pass

    def process(self, x):
        """Process data x and return a processed copy x_new."""
        pass

class BlankPreprocessor(Preprocessor):
    """Do nothing."""
    def __init__(self):
        pass

    def process(self, x):
        return x

class MeanStdPreprocessor(Preprocessor):
    """Subtract mean and normalize by standard deviation preprocessor."""
    def __init__(self, x, prev=None):
        self.prev = prev
        if prev:
            x = prev.process(x)
        self.avg = x.mean(axis=0)
        self.std = x.std(axis=0) + _SMALL_CONSTANT

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return (x - self.avg) / self.std

class StdNormPreprocessor(Preprocessor):
    """Normalize the features using standard deviation."""
    def __init__(self, x, prev=None):
        self.prev = prev
        if prev:
            x = prev.process(x)
        self.std = x.std(axis=0) + _SMALL_CONSTANT

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return x / self.std

class WhiteningPreprocessor(Preprocessor):
    """Whitening - decorrelate covariance."""
    def __init__(self, x, prev=None):
        self.prev = prev
        if prev:
            x = prev.process(x)
        self.avg = x.mean(axis=0)
        cov = x.T.dot(x) / x.shape[0]
        self.m = la.inv(la.sqrtm(cov).real + np.eye(x.shape[1]) * _SMALL_CONSTANT)

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return (x - self.avg).dot(self.m)

class PCAPreprocessor(Preprocessor):
    """PCA"""
    def __init__(self, x, prev=None, K=None):
        self.prev = prev
        if prev:
            x = prev.process(x)
        if K == None:
            K = x.shape[1] / 2 + 1
        _, self.basis, self.avg = pca(x, K)

    def process(self, x):
        if self.prev:
            x = self.prev.process(x)
        return pca_dim_reduction(x, self.basis, self.avg)



