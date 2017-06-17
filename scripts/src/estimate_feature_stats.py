#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# estimate_feature_stats.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import argparse

from sprocket.stats.gv import GV
from sprocket.stats.f0statistics import F0statistics


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('pair_ymlf', type=str,
                        help='yml file for the speaker pair')
    args = parser.parse_args()

    # estimate and save GV of orginal and target speakers
    gv = GV(args.pair_ymlf)
    gv.estimate('mcep')

    # estimate and save F0 statistics of original and target speakers
    F0stats = F0statistics(args.pair_ymlf)
    F0stats.estimate()

    return


if __name__ == '__main__':
    main()
