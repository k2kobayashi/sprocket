#!/usr/bin/python
# coding: utf-8

from sprocket import FrameByFrameVectorConverter

import numpy as np
from numpy import linalg
from sklearn.mixture import GMM
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg


class JointGMMConverter(FrameByFrameVectorConverter):

    """GMM-based frame-by-frame speech parameter mapping.

    JointGMMConverter represents a class to transform spectral features of a
    source speaker to that of a target speaker based on Gaussian Mixture Models
    of source and target joint spectral features.

    Notation
    --------
    Source speaker's feature: X = {x_t}, 0 <= t < T
    Target speaker's feature: Y = {y_t}, 0 <= t < T
    where T is the number of time frames.

    Parameters
    ----------
    gmm : sklearn.mixture.GMM
        Gaussian Mixture Models of source and target joint features

    swap : bool
        True: source -> target
        False target -> source

    ignore_0th: bool
        True: convert feature vector expect for 0th order

    Attributes
    ----------
    num_mixtures : int
        the number of Gaussian mixtures

    weights : array, shape (`num_mixtures`)
        weights of each Gaussian

    src_means : array, shape (`num_mixtures`, `order of spectral feature`)
        means of GMM for a source speaker

    tgt_means : array, shape (`num_mixtures`, `order of spectral feature`)
        means of GMM for a target speaker

    covarXX : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        variance matrix of source speaker's spectral feature

    covarXY : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        covariance matrix of source and target speaker's spectral feature

    covarYX : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        covariance matrix of target and source speaker's spectral feature

    covarYY : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        variance matrix of target speaker's spectral feature

    D : array, shape (`num_mixtures`, `order of spectral feature`,
        `order of spectral feature`)
        covariance matrices of target static spectral features

    px : sklearn.mixture.GMM
        Gaussian Mixture Models of source speaker's features

    Reference
    ---------
      - [Toda 2007] Voice Conversion Based on Maximum Likelihood Estimation
        of Spectral Parameter Trajectory.
        http://isw3.naist.jp/~tomoki/Tomoki/Journals/IEEE-Nov-2007_MLVC.pdf

    """

    def __init__(self, gmm,
                 swap=False,
                 ignore_0th=True
                 ):

        self.ignore_0th = ignore_0th

        # D is the order of spectral feature for a speaker
        self.num_mixtures, D = gmm.means_.shape[0], gmm.means_.shape[1] / 2
        self.weights = gmm.weights_

        # Split source and target parameters from joint GMM
        self.src_means = gmm.means_[:, 0:D]
        self.tgt_means = gmm.means_[:, D:]
        self.covarXX = gmm.covars_[:, :D, :D]
        self.covarXY = gmm.covars_[:, :D, D:]
        self.covarYX = gmm.covars_[:, D:, :D]
        self.covarYY = gmm.covars_[:, D:, D:]

        # swap src and target parameters
        if swap:
            self.tgt_means, self.src_means = self.src_means, self.tgt_means
            self.covarYY, self.covarXX = self.covarXX, self.covarYY
            self.covarYX, self.covarXY = self.XY, self.covarYX

        # Compute D eq.(12) in [Toda 2007]
        self.D = np.zeros(
            self.num_mixtures * D * D).reshape(self.num_mixtures, D, D)
        for m in range(self.num_mixtures):
            xx_inv_xy = np.linalg.solve(self.covarXX[m], self.covarXY[m])
            self.D[m] = self.covarYY[m] - np.dot(self.covarYX[m], xx_inv_xy)

        # p(x), which is used to compute posterior prob. for a given source
        # spectral feature in mapping stage.
        self.px = GMM(n_components=self.num_mixtures, covariance_type="full")
        self.px.means_ = self.src_means
        self.px.covars_ = self.covarXX
        self.px.weights_ = self.weights

    def get_input_shape(self):
        if self.ignore_0th:
            return self.tgt_means.shape[1] + 1

        return self.tgt_means.shape[1]

    def get_output_shape(self):
        return self.get_input_shape()

    def convert_one_frame(self, src):
        """
        Mapping source spectral feature x to target spectral feature y
        so that minimize the mean least squared error.
        More specifically, it returns the value E(p(y|x)].

        Parameters
        ----------
        src : array, shape (`order of spectral feature`)
            source speaker's spectral feature that will be transformed

        Return
        ------
        converted spectral feature
        """
        if len(src) != self.get_input_shape():
            raise Exception("Dimention mismatch")

        if self.ignore_0th:
            fv = src[1:]
        else:
            fv = src

        D = len(fv)

        # Eq.(11)
        E = np.zeros((self.num_mixtures, D))
        for m in range(self.num_mixtures):
            xx = np.linalg.solve(self.covarXX[m], fv - self.src_means[m])
            E[m] = self.tgt_means[m] + self.covarYX[m].dot(xx)

        # Eq.(9) p(m|x)
        posterior = self.px.predict_proba(np.atleast_2d(fv))

        # Eq.(13) conditinal mean E[p(y|x)]
        tgt = posterior.dot(E)

        if self.ignore_0th:
            return np.insert(tgt, 0, src[0])

        return tgt
