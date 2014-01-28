"""
This module implements some basic CRF learning.

TODO: use GP to tune model parameters.

Yujia Li, 09/2013
"""

import numpy as np
import general as gen
import time
import imgtools.pairwise as pw

def grid_search_pairwise_weights(imgs, unary, pwlist, weight_range, ground_truth):
    """
    Do a grid search to find the optimal weights for pairwise potentials.

    Here the unary potentials are fixed, and we are only learning a linear
    coefficient for each type of pairwise potentials. More specifically, the
    energy function is

        E(y) = f^u(y) + sum_k lambda_k * f_k^p(y)

    where f^u(y) is the unary potential, f_k^p(y) is the kth type of pairwise
    potentials, and lambda_k's are the weight parameters we try to learn.

    imgs is a list of images used to tune the weight parameters.
    unary is a list of unary potentials for each image.
    pwlist is a list of different types of pairwise potentials. Each element of
        the list is a pairwise potential and it is required that all the 
        pairwise potentials share the same edges and the edge weights are in 
        the same order.  The length of pwlist is actually K-1, K is the number
        of types of potentials.  The one being subtracted out is the Boykov-
        Jolly potential, which will be computed during grid search.
    weight_range is a list of lists, each sub-list is a list of possible values
        for the corresponding lambda_k. An exception is the last element in the
        list, which is actually the range for the sigma parameter of Boykov-
        Jolly pairwise potential.
    ground_truth is a matrix of ground truth labels.

    Return: (sigma, weights, acc), sigma parameter for Boykov-Jolly pairwise
        potential and a list of optimal weights and optimal accuracy.
    """
    assert (len(weight_range) == len(pwlist) + 2)
    assert (len(pwlist[0]) == ground_truth.shape[0])
    assert (len(pwlist[0]) == len(imgs))

    n_types = len(weight_range) - 1
    n_cases = ground_truth.shape[0]

    best_sigma = 0
    best_weights = [0] * n_types
    best_acc = 0

    total_trials = 1
    for i in range(n_types):
        total_trials *= len(weight_range[i])

    weights = [0] * n_types
    p = []
    for i in range(n_cases):
        p.append([pwlist[0][i][0], pwlist[0][i][1] * 0])

    t_start = time.time()

    for sigma in weight_range[n_types]:
        boykov_jolly_pw = pw.get_boykov_jolly_pw(imgs, sigma)

        for i in range(total_trials):
            # clean up the edge weights
            for j in range(n_cases):
                p[j][1] *= 0

            index = i
            for k in range(n_types):
                weights[k] = weight_range[k][index % len(weight_range[k])]
                index = index / len(weight_range[k])

                # combine pairwise potentials
                if k < n_types - 1:
                    for j in range(n_cases):
                        p[j][1] += pwlist[k][j][1] * weights[k]
                else:
                    for j in range(n_cases):
                        p[j][1] += boykov_jolly_pw[j][1] * weights[k]

            acc = gen.pixel_accuracy(
                    gen.unary_pairwise_predict(unary, p, 1), ground_truth)

            print 'sigma %4g' % sigma,
            for k in range(n_types):
                print 'lambda_%d %4g ' % (k, weights[k]),
            print 'acc %.4f ' % acc,

            if acc > best_acc:
                best_sigma = sigma
                best_weights[:] = weights
                best_acc = acc
                print '*'
            else:
                print ''

    t_end = time.time()

    print '------------------------------------------------------------------------------'
    print 'Grid search finished, best acc %.4f, best sigma %g, best lambda [' % (
            best_acc, best_sigma),
    for k in range(n_types):
        print '%g' % best_weights[k],
    print '], total time: %.2f' % (t_end - t_start)

    return best_sigma, best_weights, acc






