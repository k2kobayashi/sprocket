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

from sprocket.util.hdf5 import HDF5files
from sprocket.stats.gv import GV
from sprocket.stats.f0statistics import F0statistics


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('spkr', type=str,
                        help='Input speaker label')
    parser.add_argument('listf', type=str,
                        help='List file of the input speaker')
    parser.add_argument('h5_dir', type=str,
                        help='Hdf5 file directory of the speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Statistics directory of the speaker')
    args = parser.parse_args()

    # open h5 files
    h5s = HDF5files(args.listf, args.h5_dir)

    statspath = args.pair_dir + '/stats/' + args.spkr

    # estimate and save GV of orginal and target speakers
    gv = GV()
    gv.estimate(h5s.datalist(ext='mcep'))
    gvpath = statspath + '.gv'
    gv.save(gvpath)

    # estimate and save F0 statistics of original and target speakers
    f0stats = F0statistics()
    f0stats.estimate(h5s.datalist(ext='f0'))
    f0statspath = statspath + '.f0stats'
    f0stats.save(f0statspath)

    return


if __name__ == '__main__':
    main()
