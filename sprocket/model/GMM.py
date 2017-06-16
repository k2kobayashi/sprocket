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

import numpy as np
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
        # read w, jmu, jcov
        pass

    def save(self, fpath):
        # write as pkl file
        pass

    def train(self, jnt):
        self.param.fit(jnt)

        self.w = self.param.weights_
        self.jmu = self.param.means_
        self.jcov = self.param.covariances_

        return

    def set_conversion(self):
        # estimate parameters for conversion
        self.set_Ab()
        self.set_pX()

        return

    def convert(self, data):
        # estimate parameter sequence
        cseq, wseq, mseq, covseq = self.gmmmap(data)

        # maximum likelihood parameter generation
        odata = self.mlpg(cseq, wseq, mseq, covseq)

        return odata

    def set_Ab(self):
        # calculate A and b from self.jmu, self.jcov
        sddim = self.jmu.shape[1] // 2

        # devide GMM parameters into source and target parameters
        self.meanX = self.jmu[:, 0:sddim]
        self.meanY = self.jmu[:, sddim:]
        self.covXX = self.jcov[:, :sddim, :sddim]
        self.covXY = self.jcov[:, :sddim, sddim:]
        self.covYX = self.jcov[:, sddim:, :sddim]
        self.covYY = self.jcov[:, sddim:, sddim:]

        # calculate inverse covariance for covariance XX
        self.covXXinv = np.zeros((self.conf.n_mix, sddim, sddim))
        for m in range(self.conf.n_mix):
            self.covXXinv[m] = np.linalg.inv(self.covXX[m])

        # calculate A, b, conditional covariance
        self.A = np.zeros((self.conf.n_mix, sddim, sddim))
        self.b = np.zeros((self.conf.n_mix, sddim))
        self.cond_cov = np.zeros((self.conf.n_mix, sddim, sddim))
        for m in range(self.conf.n_mix):
            # calculate A (i.e., A = yxcov_m * xxcov_m^-1)
            self.A[m] = np.dot(self.covYX[m], self.covXXinv[m])

            # calculate b (i.e., b = mu^Y - A * mu^X)
            self.b[m] = self.meanY[m] - np.dot(self.A[m], self.meanX[m])

            # calculate conditional covariance (i.e., cov^(Y|X) = yycov - A *
            # xxcov * A')
            self.cond_cov[m] = self.covYY[
                m] - np.dot(np.dot(self.A[m], self.covXX[m, :]), self.A[m].T)
        return

    def set_pX(self):
        # probability density function for X
        self.pX = sklearn.mixture.GaussianMixture(
            n_components=self.conf.n_mix, covariance_type="full")
        self.pX.weights_ = self.w
        self.pX.means_ = self.meanX
        self.pX.covariances_ = self.covXX
        # TODO: need to estimate for calculate proba
        self.pX.precisions_cholesky_ = self.pX._compute_precision_cholesky(
           self.covXX, "full")

        None
        return

    def gmmmap(self, sddata):
        # parameter for sequencial data
        T, sddim = sddata.shape

        assert T == sddata.shape[0]

        # inference
        for t in range(T):
            print(self.pX.predict_proba(sddata[t]))

        # estimate mixture sequence
        cseq = np.argmax(wseq, axix=0)

        # conditional mean vector sequence
        mseq = np.dot(self.A, sddata) + self.b

        # conditional covariance sequence
        covseq = np.zeros((T, sddim, sddim))
        for t in range(T):
            covseq[t] = self.cond_cov[cseq[t]]

        return wseq, cseq, mseq, covseq

    def mlpg(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
