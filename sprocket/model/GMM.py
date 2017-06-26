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
import scipy.sparse
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

    def __init__(self, conf, gmm_mode=None, cv_mode='mlpg'):
        # copy parameters
        self.conf = conf
        self.gmm_mode = gmm_mode
        self.cv_mode = cv_mode

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
        self._deploy_parameters()

        # change model paramter of GMM into that of gmm_mode
        if self.gmm_mode is None:
            print('open GMM as JD-GMM of X and Y.')
        elif self.gmm_mode == 'diff':
            self._transform_gmm_into_diffgmm()
            print('open GMM as DIFFGMM.')
        elif self.gmm_mode == 'intra':
            self._transform_gmm_into_intragmm()
            raise('intra-GMM does not support now')
        else:
            raise('please choose GMM mode in [None, diff, intra]')

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
        self._deploy_parameters()
        print('GMM modeling has been done.')

        # estimate parameters for conversion
        self._set_Ab()
        self._set_pX()

        return

    def convert(self, data):
        # estimate parameter sequence
        cseq, wseq, mseq, covseq = self.gmmmap(data)

        if self.cv_mode == 'mlpg':
            # maximum likelihood parameter generation
            odata = self.mlpg(mseq, covseq)
        elif self.cv_mode == 'mmse':
            # minimum mean square error based parameter generation
            odata = self.mmse(wseq, data)
        else:
            raise('please choose conversion mode in [mlpg, mmse]')

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

    def mlpg(self, mseq, covseq):
        # parameter for sequencial data
        T, sddim = mseq.shape

        # prepare W
        W = construct_static_and_delta_matrix(T, sddim / 2)

        # prepare D
        D = get_diagonal_precision_matrix(T, sddim, covseq)

        # calculate W'D
        WD = W.T.dot(D)

        # W'DW
        WDW = WD.dot(W)

        # W'Um
        WDm = WD.dot(mseq.flatten())

        # estimate y = (W'DW)^-1 * W'Dm
        odata = scipy.sparse.linalg.spsolve(
            WDW, WDm, use_umfpack=False).reshape(T, sddim / 2)

        # return odata
        return odata

    def _deploy_parameters(self):
        # read JD-GMM parameters from self.param
        self.w = self.param.weights_
        self.jmean = self.param.means_
        self.jcov = self.param.covariances_

        # devide GMM parameters into source and target parameters
        sddim = self.jmean.shape[1] // 2
        self.meanX = self.jmean[:, 0:sddim]
        self.meanY = self.jmean[:, sddim:]
        self.covXX = self.jcov[:, :sddim, :sddim]
        self.covXY = self.jcov[:, :sddim, sddim:]
        self.covYX = self.jcov[:, sddim:, :sddim]
        self.covYY = self.jcov[:, sddim:, sddim:]

        return

    def _set_Ab(self):
        # calculate A and b from self.jmean, self.jcov
        sddim = self.jmean.shape[1] // 2

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

    def _transform_gmm_into_diffgmm(self):
        self.meanX = self.meanX
        self.meanY = self.meanY - self.meanX
        self.covXX = self.covXX
        self.covYY = self.covXX + self.covYY - self.covXY - self.covYX
        self.covXY = self.covXY - self.covXX
        self.covYX = self.covXY.transpose(0, 2, 1)

        return

    def _transform_gmm_into_intragmm(self):
        pass


def construct_static_and_delta_matrix(T, D):
    # TODO: static and delta matrix will be defined at the other place
    static = [0, 1, 0]
    delta = [-0.5, 0, 0.5]
    assert len(static) == len(delta)

    # generate full W
    DT = D * T
    ones = np.ones(DT)
    row = np.arange(2 * DT).reshape(2 * T, D)
    static_row = row[::2]
    delta_row = row[1::2]
    col = np.arange(DT)

    data = np.array([ones * static[0], ones * static[1],
                     ones * static[2], ones * delta[0],
                     ones * delta[1], ones * delta[2]]).flatten()
    row = np.array([[static_row] * 3,  [delta_row] * 3]).flatten()
    col = np.array([[col - D, col, col + D] * 2]).flatten()

    # remove component at first and end frame
    valid_idx = np.logical_not(np.logical_or(col < 0, col >= DT))

    W = scipy.sparse.csr_matrix(
        (data[valid_idx], (row[valid_idx], col[valid_idx])), shape=(2 * DT, DT))
    W.eliminate_zeros()

    return W


def get_diagonal_precision_matrix(T, D, covseq):
    return scipy.sparse.block_diag(covseq, format='csr')


def main():
    pass


if __name__ == '__main__':
    main()
