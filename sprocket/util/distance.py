#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# melcd.py
#   First ver.: 2017-06-07
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
Calculate distance between acoustic features

"""

import math
import numpy as np


def melcd(vec1, vec2):
    if len(vec1) != len(vec2):
        raise ("dimension of the vectors is different.")

    return 10.0 * math.sqrt(2 * np.sum(np.square(vec1 - vec2))) / math.log(10)


def main():
    pass


if __name__ == '__main__':
    main()
