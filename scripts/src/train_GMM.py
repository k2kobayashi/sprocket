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

import argparse

from sprocket.util.jnt import JointFeatureExtractor
from sprocket.model.GMM import GMMTrainer


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('pair_ymlf', type=str,
                        help='yml file for the speaker pair')
    args = parser.parse_args()

    # joint feature extraction
    jnt = JointFeatureExtractor(args.pair_ymlf)
    jntdata = jnt.read_jnt()

    # train GMM using joint feature vector
    GMMpath = './data/pair/clb-slt/model/GMM.pkl'
    gmm = GMMTrainer(args.pair_ymlf)
    gmm.train(jntdata)
    gmm.save(GMMpath)

    return


if __name__ == '__main__':
    main()
