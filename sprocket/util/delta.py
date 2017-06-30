#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# delta.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
calculate delta of acoustic feature sequence

"""

import numpy as np


def delta(data):
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


def main():
    pass


if __name__ == '__main__':
    main()
