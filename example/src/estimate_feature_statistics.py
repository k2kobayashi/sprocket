#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate acoustic feature statistics

"""

import argparse
import os
import sys

from sprocket.model import GV, F0statistics
from sprocket.util import HDF5

from .misc import read_feats


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
    statspath = os.path.join(args.pair_dir, 'stats', args.speaker + '.h5')
    h5 = HDF5(statspath, mode='a')

    # estimate and save F0 statistics
    f0stats = F0statistics()
    f0s = read_feats(args.list_file, h5_dir, ext='f0')
    f0stats = f0stats.estimate(f0s)
    h5.save(f0stats, ext='f0stats')
    print("f0stats save into " + statspath)

    # estimate and save GV of orginal and target speakers
    gv = GV()
    mceps = read_feats(args.list_file, h5_dir, ext='mcep')
    gvstats = gv.estimate(mceps)
    h5.save(gvstats, ext='gv')
    print("gvstats save into " + statspath)

    h5.close()


if __name__ == '__main__':
    main()
