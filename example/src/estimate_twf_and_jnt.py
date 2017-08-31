#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimate joint feature vector of the speaker pair using GMM

"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from sprocket.util.hdf5 import read_feats
from sprocket.util.jnt import JointFeatureExtractor
from yml import PairYML


def main(*argv):
    argv = argv if argv else sys.argv[1:]
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
    org_mceps = read_feats(args.org_list_file, h5_dir, ext='mcep')
    org_npows = read_feats(args.org_list_file, h5_dir, ext='npow')
    tar_mceps = read_feats(args.tar_list_file, h5_dir, ext='mcep')
    tar_npows = read_feats(args.tar_list_file, h5_dir, ext='npow')

    # extract twf and joint feature
    jnt = JointFeatureExtractor(feature='mcep',
                                jnt_iter=pconf.jnt_n_iter,
                                pairdir=args.pair_dir)
    jnt.set_GMM_parameter(n_mix=pconf.GMM_mcep_n_mix,
                          n_iter=pconf.GMM_mcep_n_iter,
                          covtype=pconf.GMM_mcep_covtype,
                          cvtype=pconf.GMM_mcep_cvtype)
    jnt.estimate(org_mceps, tar_mceps, org_npows, tar_npows)

if __name__ == '__main__':
    main()
