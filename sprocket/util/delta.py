# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.sparse


def delta(data):
    """Calculate delta component

    Parameters
    ----------
    data : array, shape (`T`, `dim`)
        Array of static matrix sequence.

    Returns
    -------
    delta : array, shape (`T`, `dim`)
        Array delta matrix sequence.

    """

    if data.ndim == 1:
        # change vector into 1d-array
        T = len(data)
        dim = data.ndim
        data = data.reshape(T, dim)
    else:
        T, dim = data.shape

    win = np.array([-0.5, 0, 0.5], dtype=np.float64)
    delta = np.zeros((T, dim))

    delta[0] = 0.5 * data[1]
    delta[-1] = - 0.5 * data[-2]

    for i in range(len(win)):
        delta[1:T - 1] += win[i] * data[i:T - 2 + i]

    return delta


def construct_static_and_delta_matrix(T, D):
    """Calculate static and delta transformation matrix

    Parameters
    ----------
    T : scala, `T`
        Scala of time length

    D : scala, `D`
        Scala of the number of dimentsion

    Returns
    -------
    W : array, shape (`2 * D * T`, `D * T`)
        Array of Static and delta transformation matrix.

    """

    # TODO: static and delta matrix will be defined at the other place
    static = [0, 1, 0]
    delta = [-0.5, 0, 0.5]
    assert len(static) == len(delta)

    # generate full W
    DT = D * T
    ones = np.ones(DT)
    row = np.arange(2 * DT).reshape(2 * T, D)
    static_row = row[::2]
    delta_row = row[1::2]
    col = np.arange(DT)

    data = np.array([ones * static[0], ones * static[1],
                     ones * static[2], ones * delta[0],
                     ones * delta[1], ones * delta[2]]).flatten()
    row = np.array([[static_row] * 3,  [delta_row] * 3]).flatten()
    col = np.array([[col - D, col, col + D] * 2]).flatten()

    # remove component at first and end frame
    valid_idx = np.logical_not(np.logical_or(col < 0, col >= DT))

    W = scipy.sparse.csr_matrix(
        (data[valid_idx], (row[valid_idx], col[valid_idx])), shape=(2 * DT, DT))
    W.eliminate_zeros()

    return W
