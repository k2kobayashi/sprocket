#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# estimate_jnt.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
estimate joint feature vector of the speaker pair using GMM

"""

import argparse

from sprocket.util.hdf5 import HDF5files
from sprocket.util.yml import PairYML
from sprocket.util.jnt import JointFeatureExtractor


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org_list_file', type=str,
                        help='List file of original speaker')
    parser.add_argument('tar_list_file', type=str,
                        help='List file of target speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    parser.add_argument('h5_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args()

    # read pair-dependent yml file
    # conf = PairYML(args.pair_ymlf)

    orgh5s = HDF5files(args.org_list_file, args.h5_dir)
    tarh5s = HDF5files(args.tar_list_file, args.h5_dir)

    # joint feature extraction
    jnt = JointFeatureExtractor(
        feature='mcep',
        n_iter=3,
        pairdir=args.pair_dir)
    jnt.estimate(
        orgh5s.datalist('mcep'),
        tarh5s.datalist('mcep'),
        orgh5s.datalist('npow'),
        tarh5s.datalist('npow'))

    return

if __name__ == '__main__':
    main()
