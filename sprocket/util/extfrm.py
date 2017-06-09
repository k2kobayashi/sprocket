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

def extfrm(npow, mcep):
    T = len(npow)
    threshold = -20
    for t in range (T):
        if npow[t] < threshold:
            # remove feature
            pass
        else:
            pass
            # concatenate
    return extmcep


def main():
    pass


if __name__ == '__main__':
    main()
