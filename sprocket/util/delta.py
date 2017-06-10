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
    T, dim = data.shape
    win = [-0.5, 0, 0.5]

    delta = np.zeros((T, dim))
    delta[0] = 0.5 * data[1]
    delta[T] = - 0.5 * data[T - 1]
    for t in range(1, T - 1):
        delta[t] = np.dot(data[t - 1:t + 1], win)

    return delta


def main():
    pass


if __name__ == '__main__':
    main()
