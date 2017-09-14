#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimate acoustic feature statistics

"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from .misc import read_feats
from sprocket.stats.f0statistics import F0statistics
from sprocket.stats.gv import GV


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('speaker', type=str,
                        help='Input speaker label')
    parser.add_argument('list_file', type=str,
                        help='List file of the input speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Statistics directory of the speaker')
    args = parser.parse_args(argv)

    # open h5 files
    h5_dir = os.path.join(args.pair_dir, 'h5')
    statspath = os.path.join(args.pair_dir, 'stats', args.speaker)

    # estimate and save GV of orginal and target speakers
    gv = GV()
    mceps = read_feats(args.list_file, h5_dir, ext='mcep')
    gv.estimate(mceps)
    gvpath = os.path.join(statspath + '.gv')
    gv.save(gvpath)
    print(gvpath)

    # estimate and save F0 statistics of original and target speakers
    f0stats = F0statistics()
    f0s = read_feats(args.list_file, h5_dir, ext='f0')
    f0stats.estimate(f0s)
    f0statspath = os.path.join(statspath + '.f0stats')
    f0stats.save(f0statspath)
    print(f0statspath)


if __name__ == '__main__':
    main()
