#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# estimate_feature_statistics.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
Estimate acoustic feature statistics


"""

from __future__ import division, print_function, absolute_import

import os
import argparse

from sprocket.util.hdf5 import HDF5files
from sprocket.stats.gv import GV
from sprocket.stats.f0statistics import F0statistics


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('speaker', type=str,
                        help='Input speaker label')
    parser.add_argument('list_file', type=str,
                        help='List file of the input speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Statistics directory of the speaker')
    args = parser.parse_args()

    # open h5 files
    h5_dir = os.path.join(args.pair_dir, 'h5')
    h5s = HDF5files(args.list_file, h5_dir)

    statspath = os.path.join(args.pair_dir, 'stats', args.speaker)

    # estimate and save GV of orginal and target speakers
    gv = GV()
    gv.estimate(h5s.datalist(ext='mcep'))
    gvpath = os.path.join(statspath + '.gv')
    gv.save(gvpath)
    print(gvpath)

    # estimate and save F0 statistics of original and target speakers
    f0stats = F0statistics()
    f0stats.estimate(h5s.datalist(ext='f0'))
    f0statspath = os.path.join(statspath + '.f0stats')
    f0stats.save(f0statspath)
    print(f0statspath)


if __name__ == '__main__':
    main()
