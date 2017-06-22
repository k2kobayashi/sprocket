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

import os
import numpy as np
import sklearn.mixture
from sklearn.externals import joblib
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky


class GMMTrainer(object):

    """
    A GMM trainer

    This class assumes:

    TODO:
    Training and converting acoustic feature

    Attributes
    ---------
    conf: parameters read from speaker yml file
    param: parameters of the GMM
    w: weigh of the GMM
    jmean: mean vector of the GMM
    jcov: covariance matrix of the GMM

    """

    def __init__(self, conf, diff=False):
        # copy parameters
        self.conf = conf
        self.diff = diff

        # parameter definition
        self.param = sklearn.mixture.GaussianMixture(
            n_components=self.conf.n_mix,
            covariance_type=self.conf.covtype,
            max_iter=self.conf.n_iter)

    def open(self, fpath):
        # read GMM from pkl file
        if not os.path.exists(fpath):
            raise('pkl file of GMM does not exists.')
        # read model parameter file
        self.param = joblib.load(fpath)

        # read joint model parameters
        self.w = self.param.weights_
        self.jmean = self.param.means_
        self.jcov = self.param.covariances_

        # estimate parameters for conversion
        self._set_Ab()
        self._set_pX()

        print('open GMM has been done.')
        return

    def save(self, fpath):
        # save GMM intp pkl file
        dirname = os.path.dirname(fpath)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # save model parameter file
        joblib.dump(self.param, fpath)

    def train(self, jnt):
        print('GMM modeling starts')
        self.param.fit(jnt)
        print('GMM modeling has been done.')

        self.w = self.param.weights_
        self.jmean = self.param.means_
        self.jcov = self.param.covariances_

        # estimate parameters for conversion
        self._set_Ab()
        self._set_pX()

        return

    def convert(self, data):
        # estimate parameter sequence
        cseq, wseq, mseq, covseq = self.gmmmap(data)

        # minimum mean square error based parameter generation
        odata = self.mmse(wseq, data)

        # TODO # maximum likelihood parameter generation
        # odata = self.mlpg(cseq, wseq, mseq, covseq)

        return odata

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
            # read maximum likelihood mixture component in frame t
            m = cseq[t]

            # conditional mean vector sequence
            mseq[t] = self.meanY[m] + \
                np.dot(self.A[m], sddata[t] - self.meanX[m])

            # conditional covariance sequence
            covseq[t] = self.cond_cov_inv[m]

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

        # retern static and throw away delta component
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

    def _set_Ab(self):
        # calculate A and b from self.jmean, self.jcov
        sddim = self.jmean.shape[1] // 2

        # devide GMM parameters into source and target parameters
        self.meanX = self.jmean[:, 0:sddim]
        self.meanY = self.jmean[:, sddim:]
        self.covXX = self.jcov[:, :sddim, :sddim]
        self.covXY = self.jcov[:, :sddim, sddim:]
        self.covYX = self.jcov[:, sddim:, :sddim]
        self.covYY = self.jcov[:, sddim:, sddim:]

        # Convert to Diff-GMM
        if self.diff:
            self.meanX = self.meanX
            self.meanY = self.meanY - self.meanX
            covXY = self.covXY.copy()
            covYX = self.covYX.copy()
            covYY = self.covYY.copy()
            self.covXX = self.covXX
            self.covYY = self.covXX + covYY - covXY - covYX
            self.covXY = covXY - self.covXX
            self.covYX = self.covXY.transpose(0, 2, 1)

        # calculate inverse covariance for covariance XX in each mixture
        self.covXXinv = np.zeros((self.conf.n_mix, sddim, sddim))
        for m in range(self.conf.n_mix):
            self.covXXinv[m] = np.linalg.inv(self.covXX[m])

        # calculate A, b, and conditional covariance given X
        self.A = np.zeros((self.conf.n_mix, sddim, sddim))
        self.b = np.zeros((self.conf.n_mix, sddim))
        self.cond_cov_inv = np.zeros((self.conf.n_mix, sddim, sddim))
        for m in range(self.conf.n_mix):
            # calculate A (i.e., A = yxcov_m * xxcov_m^-1)
            self.A[m] = np.dot(self.covYX[m], self.covXXinv[m])

            # calculate b (i.e., b = mean^Y - A * mean^X)
            self.b[m] = self.meanY[m] - np.dot(self.A[m], self.meanX[m])

            # calculate conditional covariance
            # (i.e., cov^(Y|X)^-1 = (yycov - A * xycov)^-1)
            self.cond_cov_inv[m] = np.linalg.inv(self.covYY[
                m] - np.dot(self.A[m], self.covXY[m]))

        return

    def _set_pX(self):
        # probability density function of X
        self.pX = sklearn.mixture.GaussianMixture(
            n_components=self.conf.n_mix, covariance_type=self.conf.covtype)
        self.pX.weights_ = self.w
        self.pX.means_ = self.meanX
        self.pX.covariances_ = self.covXX

        # following function is required to estimate porsterior
        # P(X | \lambda^(X)))
        self.pX.precisions_cholesky_ = _compute_precision_cholesky(
            self.covXX, self.conf.covtype)

        return


def main():
    pass


if __name__ == '__main__':
    main()
