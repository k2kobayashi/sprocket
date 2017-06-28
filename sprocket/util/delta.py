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

from sprocket.util.decorators import optional_jit


@optional_jit
def delta(data):
    T, dim = data.shape
    win = np.array([-0.5, 0, 0.5], dtype=np.float64)

    delta = np.zeros((T, dim))

    delta[0] = 0.5 * data[1]
    delta[-1] = - 0.5 * data[-2]
    for t in range(1, T - 1):
        # print(t - 1, t, t + 1, t + 2,
        #       data[t - 1:t + 1].shape, data[t - 1:t + 2].shape)
        delta[t] = np.dot(data[t - 1:t + 2].T, win)

    return delta


def main():
    pass


if __name__ == '__main__':
    main()
