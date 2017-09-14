#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
train GMM based on joint feature vector

"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
from sklearn.externals import joblib

from sprocket.model.GMM import GMMTrainer
from sprocket.util.hdf5 import HDF5
from yml import PairYML


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args(argv)

    # read pair-dependent yml file
    pconf = PairYML(args.pair_yml)

    # read joint feature vector
    jntf = os.path.join(args.pair_dir, 'jnt',
                        'it' + str(pconf.jnt_n_iter + 1) + '_jnt.h5')
    h5 = HDF5(jntf, mode='r')
    jnt = h5.read(ext='jnt')

    # train GMM for mcep using joint feature vector
    gmm = GMMTrainer(n_mix=pconf.GMM_mcep_n_mix, n_iter=pconf.GMM_mcep_n_iter,
                     covtype=pconf.GMM_mcep_covtype)
    gmm.train(jnt)

    # save GMM
    gmm_dir = os.path.join(args.pair_dir, 'model')
    os.makedirs(gmm_dir, exist_ok=True)
    gmmpath = os.path.join(gmm_dir, 'GMM.pkl')
    joblib.dump(gmm.param, gmmpath)
    print(gmmpath)


if __name__ == '__main__':
    main()
