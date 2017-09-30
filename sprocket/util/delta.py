# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse


def delta(data, win=[-1.0, 1.0, 0]):
    """Calculate delta component

    Parameters
    ----------
    data : array, shape (`T`, `dim`)
        Array of static matrix sequence.
    win: array, optional, shape (`3`)
        The shape of window matrix.
        Default set to [-1.0, 1.0, 0].

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

    win = np.array(win, dtype=np.float64)
    delta = np.zeros((T, dim))

    delta[0] = win[0] * data[0] + win[1] * data[1]
    delta[-1] = win[0] * data[-2] + win[1] * data[-1]

    for i in range(len(win)):
        delta[1:T - 1] += win[i] * data[i:T - 2 + i]

    return delta


def static_delta(data, win=[-1.0, 1.0, 0]):
    """Calculate static and delta component

    Parameters
    ----------
    data : array, shape (`T`, `dim`)
        Array of static matrix sequence.
    win: array, optional, shape (`3`)
        The shape of window matrix.
        Default set to [-1.0, 1.0, 0].

    Returns
    -------
    sddata: array, shape (`T`, `dim*2`)
        Array static and delta matrix sequence.

    """

    sddata = np.c_[data, delta(data, win)]
    return sddata


def construct_static_and_delta_matrix(T, D, win=[-1.0, 1.0, 0]):
    """Calculate static and delta transformation matrix

    Parameters
    ----------
    T : scala, `T`
        Scala of time length
    D : scala, `D`
        Scala of the number of dimentsion
    win: array, optional, shape (`3`)
        The shape of window matrix for delta.
        Default set to [-1.0, 1.0, 0].

    Returns
    -------
    W : array, shape (`2 * D * T`, `D * T`)
        Array of static and delta transformation matrix.

    """

    static = [0, 1, 0]
    delta = win
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
