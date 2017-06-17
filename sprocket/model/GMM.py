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

from sklearn.externals import joblib
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky


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
        # read model parameter file
        self.param = joblib.load(fpath)

    def save(self, fpath):
        # save model parameter file
        joblib.dump(self.param, fpath)

    def train(self, jnt):
        print('GMM modeling starts')
        self.param.fit(jnt)
        print('GMM modeling has been done.')

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

        # minimum mean square error based parameter generation
        odata = self.mmse(wseq, data)

        # TODO # maximum likelihood parameter generation
        # odata = self.mlpg(cseq, wseq, mseq, covseq)

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
        self.cond_cov_inv = np.zeros((self.conf.n_mix, sddim, sddim))
        for m in range(self.conf.n_mix):
            # calculate A (i.e., A = yxcov_m * xxcov_m^-1)
            self.A[m] = np.dot(self.covYX[m], self.covXXinv[m])

            # calculate b (i.e., b = mu^Y - A * mu^X)
            self.b[m] = self.meanY[m] - np.dot(self.A[m], self.meanX[m])

            # calculate conditional covariance
            # (i.e., cov^(Y|X)^-1 = (yycov - A * xycov)^-1)
            self.cond_cov_inv[m] = np.linalg.inv(self.covYY[
                m] - np.dot(self.A[m], self.covXY[m]))

        return

    def set_pX(self):
        # probability density function for X
        self.pX = sklearn.mixture.GaussianMixture(
            n_components=self.conf.n_mix, covariance_type=self.conf.covtype)
        self.pX.weights_ = self.w
        self.pX.means_ = self.meanX
        self.pX.covariances_ = self.covXX
        # this function is required to estimate porsterior  P(X | \lambda^(X)))
        self.pX.precisions_cholesky_ = _compute_precision_cholesky(
            self.covXX, self.conf.covtype)

        return

    def gmmmap(self, sddata):
        # parameter for sequencial data
        T, sddim = sddata.shape

        # estimate posterior sequence
        wseq = self.pX.predict_proba(sddata)

        # estimate mixture sequence
        cseq = np.argmax(wseq, axis=1)

        mseq = np.zeros((T, sddim))
        covseq = np.zeros((T, sddim, sddim))
        for t in range(T):
            m = cseq[t]
            # conditional mean vector sequence
            mseq[t] = self.meanY[m] + \
                np.dot(self.A[m], sddata[t] - self.meanX[m])

            # conditional covariance sequence
            covseq[t] = self.cond_cov_inv[cseq[t]]

        return cseq, wseq, mseq, covseq

    def mmse(self, wseq, sddata):
        # parameter for sequencial data
        T, sddim = sddata.shape

        odata = np.zeros((T, sddim))
        for t in range(T):
            for m in range(self.conf.n_mix):
                odata[t] += wseq[t, m] * \
                    (self.meanY[m] +
                     np.dot(self.A[m], sddata[t] - self.meanX[m]))

        return odata[:, :sddim / 2]

    def mlpg(self, wseq, cseq, mseq, covseq):
        pass
        # # TODO parameter for sequencial data
        # T, sddim = mseq.shape

        # # prepare W

        # # prepare U

        # # estimate W'u

        # # W'UW
        # WUW =

        # # W'Um
        # WUm =

        # # calculate (W'UW)^-1 * W'UM
        # odata = np.dot(np.linalg.inv(WUW), WUm)

        # return odata


def main():
    pass


if __name__ == '__main__':
    main()
