#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# extfrm.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import numpy as np


def extfrm(npow, data):
    threshold = -20
    T = data.shape[0]
    if len(npow) != T:
        raise("Length of two vectors is different.")

    index = np.where(npow > threshold)
    return data[index]


def main():
    pass


if __name__ == '__main__':
    main()
