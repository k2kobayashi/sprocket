#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# GMM.py
#   First ver.: 2017-06-15
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import sklearn.mixture
from sprocket.util.yml import PairYML


class GMMTrainer(object):

    def __init__(self, yml):
        # read pair-dependent yml file
        self.conf = PairYML(yml)

        # parameter definition
        self.param = sklearn.mixture.GaussianMixture(
            n_components=self.conf.n_mix,
            covariance_type=self.conf.covtype,
            max_iter=self.conf.n_iter)

    def open(self, fpath):
        pass

    def save(self, fpath):
        pass

    def train(self, jnt):
        self.param.fit(jnt)

        self.w = self.param.weights_
        self.mu = self.param.means_
        self.cov = self.param.covariances_

        return

    def convert(self, data):
        self.set_Ab()

        # estimate parameter sequence
        cseq, wseq, mseq, covseq = self.gmmmap(data)

        # parameter generation
        odata = self.mlpg(cseq, wseq, mseq, covseq)

        return odata

    def set_Ab(self):
        # calculate A and b
        pass


    def gmmmap(self, data):
        # estimate mixture sequence

        # conditional mean vector sequence

        # conditional covariance sequence
        pass

    def mlpg(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
