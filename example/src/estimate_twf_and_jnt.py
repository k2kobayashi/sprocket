#! /usr/bin/env python
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

from __future__ import division, print_function, absolute_import

import os
import argparse

from sprocket.util.hdf5 import HDF5files
from sprocket.util.jnt import JointFeatureExtractor

from yml import PairYML


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('org_list_file', type=str,
                        help='List file of original speaker')
    parser.add_argument('tar_list_file', type=str,
                        help='List file of target speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args()

    # read pair-dependent yml file
    pconf = PairYML(args.pair_yml)

    h5_dir = os.path.join(args.pair_dir, 'h5')
    org_h5s = HDF5files(args.org_list_file, h5_dir)
    tar_h5s = HDF5files(args.tar_list_file, h5_dir)

    # extract twf and joint feature
    jnt = JointFeatureExtractor(feature='mcep',
                                jnt_iter=pconf.jnt_n_iter,
                                pairdir=args.pair_dir)
    jnt.set_GMM_parameter(n_mix=pconf.GMM_mcep_n_mix,
                          n_iter=pconf.GMM_mcep_n_iter,
                          covtype=pconf.GMM_mcep_covtype,
                          cvtype=pconf.GMM_mcep_cvtype)
    jnt.estimate(org_h5s.datalist('mcep'), tar_h5s.datalist('mcep'),
                 org_h5s.datalist('npow'), tar_h5s.datalist('npow'))


if __name__ == '__main__':
    main()
