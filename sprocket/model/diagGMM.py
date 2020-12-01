# -*- coding: utf-8 -*-

import numpy as np
import sklearn.mixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky


class BlockDiagonalGaussianMixture(sklearn.mixture.GaussianMixture):
    """GMM with block diagonal covariance matrix
    This class offers the training of GMM with block diagonal covariance matrix.
    Note that the parent class (GaussianMixture) is trained as full-covariance matrix

    Parameters
    ----------
    n_mix : int, optional
        The number of mixture components of the GMM
        Default set to 32.
    n_iter : int, optional
        The number of iteration for EM algorithm.
        Default set to 100.
    floor : str, optional
        Flooring of covariance matrix

    Attributes
    ----------
    param :
        Sklean-based model parameters of the GMM
    """

    def __init__(self, n_mix=32, n_iter=100, floor=1e-6):
        super().__init__(
            n_components=n_mix, reg_covar=floor, max_iter=n_iter, covariance_type="full"
        )
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.floor = floor

        # seed for random in sklearn
        self.random_state = np.random.mtrand._rand

    def fit(self, X):
        """Fit GMM parameters to X

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        """
        # initialize
        self._initialize_parameters(X, self.random_state)
        lower_bound = -np.infty

        for n in range(self.n_iter):
            # E-step
            log_prob_norm, log_resp = self._e_step(X)

            # M-step
            self._m_step(X, log_resp)

            # check convergence
            back_lower_bound = lower_bound
            lower_bound = self._compute_lower_bound(log_resp, log_prob_norm)

    def _m_step(self, X, log_resp):
        """M step.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        log_resp : array-like, shape (n_samples, n_components)
            Logarithm of the posterior probabilities (or responsibilities) of
            the point of each sample in X.
        """
        n_samples, _ = X.shape
        (
            self.weights_,
            self.means_,
            self.covariances_,
        ) = self._estimate_gaussian_parameters(
            X, np.exp(log_resp), self.reg_covar, self.covariance_type
        )
        self.weights_ /= n_samples

        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type
        )

    def _estimate_gaussian_parameters(self, X, resp, reg_covar, covariance_type):
        """Estimate the Gaussian distribution parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input data array.

        resp : array-like, shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        reg_covar : float
            The regularization added to the diagonal of the covariance matrices.

        covariance_type : {'full', 'tied', 'diag', 'spherical'}
            The type of precision matrices.

        Returns
        -------
        nk : array-like, shape (n_components,)
            The numbers of data samples in the current components.

        means : array-like, shape (n_components, n_features)
            The centers of the current components.

        covariances : array-like (n_components, n_features, n_features)
            The covariance matrix of the current components.
            The shape depends of the covariance_type.
        """
        # estimate weight and mean
        nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
        means = np.dot(resp.T, X) / nk[:, np.newaxis]

        # estimate covariance
        n_components, n_features = means.shape
        diagcov = self._calculate_diag_covariances(resp, nk, X, X, means, means)
        xycov = self._calculate_diag_covariances(
            resp,
            nk,
            X[:, : n_features // 2],
            X[:, n_features // 2 :],
            means[:, : n_features // 2],
            means[:, n_features // 2 :],
        )
        # block_diag to full
        covariances = self._block_diag_to_full(diagcov, xycov)
        return nk, means, covariances

    def _block_diag_to_full(self, diagcov, xycov):
        """Transform diagonal covariance to full covariance

        Parameters
        ----------
        diagcov : array-like, shape (n_components, n_features)
            Diagonal covariance

        xycov : array-like, shape (n_components, n_features // 2)
            Variance-covariance

        Returns
        -------
        covariance : array-like, shape (n_components, n_features, n_features)
            Full covariance consiting of xxcov, xycov, yxcov, yycov
        """
        n_components, n_features = diagcov.shape
        covariances = np.empty((n_components, n_features, n_features))
        for m in range(n_components):
            covariances[m] = np.diag(diagcov[m])
            covariances[m, n_features // 2 :, : n_features // 2] = np.diag(xycov[m])
            covariances[m, : n_features // 2, n_features // 2 :] = np.diag(xycov[m])
        return covariances

    def _calculate_diag_covariances(self, resp, nk, x, y, xmeans, ymeans):
        """Calculate diagonal covariance in each portion

        Parameters
        ----------
        resp : array-like, shape (n_samples, n_components)
            The responsibilities for each data sample in X.

        nk : array-like, shape (n_components,)
            The numbers of data samples in the current components.

        x, y : array-like, shape (n_samples, n_features)
            The input data array of source and atarget.

        xmeans, ymeans : array-like, shape (n_components, n_features)
            Mean of x and y

        Returns
        -------
        diag_covariances : array-like, shape (n_components, n_features)
        """
        avg_XY = np.dot(resp.T, x * y) / nk[:, np.newaxis]
        avg_xymeans = xmeans * ymeans
        avg_x_ymeans = ymeans * np.dot(resp.T, x) / nk[:, np.newaxis]
        avg_y_xmeans = xmeans * np.dot(resp.T, y) / nk[:, np.newaxis]
        diag_covariances = (
            avg_XY - (avg_x_ymeans + avg_y_xmeans) + avg_xymeans + self.floor
        )
        return diag_covariances
