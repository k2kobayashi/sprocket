#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# train_GMM.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
train GMM based on joint feature vector

"""

import os
import argparse

from sprocket.model.GMM import GMMTrainer
from sprocket.util.hdf5 import HDF5

from yml import PairYML


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args()

    # read pair-dependent yml file
    pconf = PairYML(args.pair_yml)

    # read joint feature vector
    jntf = os.path.join(args.pair_dir, 'jnt',
                        'it' + str(pconf.n_jntiter) + '.h5')
    h5 = HDF5(jntf, mode='r')
    jnt = h5.read(ext='jnt')

    # train GMM using joint feature vector
    model_dir = os.path.join(args.pair_dir, 'model')
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    GMMpath = os.path.join(model_dir, 'GMM.pkl')
    gmm = GMMTrainer(n_mix=pconf.n_mix, n_iter=pconf.n_iter,
                     covtype=pconf.covtype)
    gmm.train(jnt)
    gmm.save(GMMpath)

    return


if __name__ == '__main__':
    main()
