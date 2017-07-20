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


"""

import os
import argparse

from sprocket.model.GMM import GMMTrainer
from sprocket.util.hdf5 import HDF5


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('pair_dir', type=str,
                        help='yml file for the speaker pair')
    args = parser.parse_args()

    # read joint feature vector
    jntf = args.pair_dir + '/jnt/it3.h5'
    h5 = HDF5(jntf, mode='r')
    jnt = h5.read(ext='mat')

    # train GMM using joint feature vector
    GMMpath = os.path.join(args.pair_dir + '/model/GMM.pkl')
    gmm = GMMTrainer()
    gmm.train(jnt)
    gmm.save(GMMpath)

    return


if __name__ == '__main__':
    main()
