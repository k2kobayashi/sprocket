# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse
import sklearn.mixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky

from sprocket.util.delta import construct_static_and_delta_matrix


class BlockDiagonalGaussianMixture(sklearn.mixture.GaussianMixture):
    """GMM with block diagonal covariance matrix
    This class offers the training of GMM with block diagonal covariance matrix.
    Note that the parent class (GaussianMixture) is trained as full-covariance matrix

    Parameters
    ---------
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
        super().__init__(n_components=n_mix, reg_covar=floor, max_iter=n_iter,
                         covariance_type='full')
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.floor = floor

        # seed for random in sklearn
        self.random_state = np.random.mtrand._rand

    def fit(self, jnt):
        # create mask
        blockdiag_mask = self._generate_blockdiag_mask(jnt.shape[1])

        # initialize
        self._initialize_parameters(jnt, self.random_state)
        lower_bound = -np.infty

        for n in range(self.n_iter):
            # EM algorithm
            log_prob_norm, log_resp = self._e_step(jnt)
            self._m_step(jnt, log_resp)

            self.covariances_ += self.floor

            # apply block diagonal mask to covariance
            self._apply_blockdiag_mask(blockdiag_mask)

            # check converge
            back_lower_bound = lower_bound
            lower_bound = self._compute_lower_bound(
                log_resp, log_prob_norm)

    def _generate_blockdiag_mask(self, jdim):
        """Generate a mask to make zero excluding diagonal components

        """
        sddim = jdim // 2
        mask = scipy.sparse.diags(
            [1, 1, 1], [-sddim, 0, sddim], shape=(jdim, jdim)).toarray()
        return mask

    def _apply_blockdiag_mask(self, mask):
        """Applying diagonal mask to full-covariance matrix

        """
        for n in range(self.n_mix):
            self.covariances_[n] *= mask

    def _mstep(self, jnt, log_resp):
        """M-step
        M step should be implemented by ourself because the m-step for full-covariance
        matrix is too slow.

        """
        raise NotImplementedError()


class GMMTrainer(object):

    """GMM trainer
    This class offers the training of GMM with several types of covariance matrix.

    Parameters
    ---------
    n_mix : int, optional
        The number of mixture components of the GMM
        Default set to 32.
    n_iter : int, optional
        THe number of iteration for EM algorithm.
        Default set to 100.
    covtype : str, optional
        The type of covariance matrix of the GMM
        'full': full-covariance matrix
        'block_diag' : block-diagonal matrix

    Attributes
    ----------
    param :
        Sklean-based model parameters of the GMM

    """

    def __init__(self, n_mix=32, n_iter=100, covtype='full'):
        self.n_mix = n_mix
        self.n_iter = n_iter
        self.covtype = covtype

        self.random_state = np.random.mtrand._rand

        # construct GMM parameter
        if self.covtype == 'full':
            self.param = sklearn.mixture.GaussianMixture(
                n_components=self.n_mix,
                covariance_type=self.covtype,
                max_iter=self.n_iter)
        elif self.covtype == 'block_diag':
            self.param = BlockDiagonalGaussianMixture(
                n_mix=self.n_mix,
                n_iter=self.n_iter)
        else:
            raise ValueError('Covariance type should be full or block_diag')

    def open_from_param(self, param):
        """Open GMM from sklearn.GaussianMixture

        Parameters
        ---------
        trainer: GMMTrainer
            GMMTrainer class

        """
        self.param = param
        return

    def train(self, jnt):
        """Fit GMM parameter from given joint feature vector

        Parameters
        ---------
        jnt : array, shape(`T`, `dim`)
            Joint feature vector of original and target feature vector consisting
            of static and delta components

        """
        self.param.fit(jnt)

    def estimate_responsibility(self, ref_jnt):
        """E-step for the single-path training

        Parameters
        ----------
        ref_jnt: array, shape(`T`, `ref_dim`)
            Reference joint feature vector of original and target feature vector consisting
            of static and delta components, which was already fit.
        """
        if self.param is None:
            raise ValueError(
                'Please load param before call estimate_responsibility')

        # perform estep
        _, self.log_resp = self.param._e_step(ref_jnt)

    def train_singlepath(self, tar_jnt):
        """Fit GMM parameter based on single-path training
        M-step :
            Update GMM parameter using `self.log_resp`, and `tar_jnt`

        Parameters
        ----------
        tar_jnt: array, shape(`T`, `tar_dim`)
            Joint feature vector of original and target feature vector consisting
            of static and delta components, which will be modeled.

        Returns
        -------
        param :
            Sklean-based model parameters of the GMM
        """

        if self.covtype == 'full':
            single_param = sklearn.mixture.GaussianMixture(
                n_components=self.n_mix,
                covariance_type=self.covtype,
                max_iter=1)
        elif self.covtype == 'block_diag':
            single_param = BlockDiagonalGaussianMixture(
                n_mix=self.n_mix,
                n_iter=self.n_iter)
        else:
            raise ValueError('Covariance type should be full or block_diag')

        # initialize target single-path param
        single_param._initialize_parameters(tar_jnt, self.random_state)

        # perform mstep
        single_param._m_step(tar_jnt, self.log_resp)

        if self.covtype == 'block_diag':
            blockdiag_mask = single_param._generate_blockdiag_mask(tar_jnt.shape[1])
            single_param._apply_blockdiag_mask(blockdiag_mask)

        return single_param


class GMMConvertor(object):
    """A GMM Convertor
    This class offers the several conversion techniques such as Maximum Likelihood
    Parameter Generation (MLPG) and Mimimum Mean Square Error (MMSE).
    Note that the conversion is performed while regarding GMM covariance
    as full-covariance matrix

    Parameters
    ----------
    n_mix : int, optional
        The number of mixture components of the GMM
        Default set to 32.
    gmmmode: str, optional
        The type of the GMM for opening
        `None` : Normal JD-GMM
        `diff` : Differential GMM
        `intra` : Intra-speaker GMM

    Attributes
    ----------
    param :
        Sklean-based model parameters of the GMM
    w : shape (`n_mix`)
        Vector of mixture component weight of the GMM
    jmean : shape (`n_mix`, `jnt.shape[0]`)
        Array of joint mean vector of the GMM
    jcov: shape (`n_mix`, `jnt.shape[0]`, `jnt.shape[0]`)
        Array of joint covariance matrix of the GMM

    """

    def __init__(self, n_mix=32, covtype='full', gmmmode=None):
        self.n_mix = n_mix
        self.gmmmode = gmmmode

    def open_from_param(self, param):
        """Open GMM from GMMTrainer

        Parameters
        ---------
        trainer: GMMTrainer
            GMMTrainer class

        """

        self.param = param
        self._deploy_parameters()

        return

    def convert(self, data, cvtype='mlpg'):
        """Convert data based on conditional probability densify function

        Parameters
        ---------
        data : array, shape(`T`, `dim`)
            Original data will be converted
        cvtype: str, optional
            Type of conversion technique
            `mlpg` : maximum likelihood parameter generation
            `mmse` : minimum mean square error

        Returns
        ---------
        odata : array, shape(`T`, `dim`)
            Converted data

        """

        # estimate parameter sequence
        cseq, wseq, mseq, covseq = self._gmmmap(data)

        if cvtype == 'mlpg':
            # maximum likelihood parameter generation
            odata = self._mlpg(mseq, covseq)
        elif cvtype == 'mmse':
            # minimum mean square error based parameter generation
            odata = self._mmse(wseq, data)
        else:
            raise ValueError('please choose conversion mode in `mlpg`, `mmse`')

        return odata

    def _gmmmap(self, sddata):
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
                self.A[m] @ (sddata[t] - self.meanX[m])

            # conditional covariance sequence
            covseq[t] = self.cond_cov_inv[m]

        return cseq, wseq, mseq, covseq

    def _mmse(self, wseq, sddata):
        # parameter for sequencial data
        T, sddim = sddata.shape

        odata = np.zeros((T, sddim))
        for t in range(T):
            for m in range(self.n_mix):
                odata[t] += wseq[t, m] * \
                    (self.meanY[m] +
                     self.A[m] @ (sddata[t] - self.meanX[m]))

        # retern static and throw away delta component
        return odata[:, :sddim // 2]

    def _mlpg(self, mseq, covseq):
        # parameter for sequencial data
        T, sddim = mseq.shape

        # prepare W
        W = construct_static_and_delta_matrix(T, sddim // 2)

        # prepare D
        D = get_diagonal_precision_matrix(T, sddim, covseq)

        # calculate W'D
        WD = W.T @ D

        # W'DW
        WDW = WD @ W

        # W'Um
        WDm = WD @ mseq.flatten()

        # estimate y = (W'DW)^-1 * W'Dm
        odata = scipy.sparse.linalg.spsolve(
            WDW, WDm, use_umfpack=False).reshape(T, sddim // 2)

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

        # change model paramter of GMM into that of gmmmode
        if self.gmmmode is None:
            pass
        elif self.gmmmode == 'diff':
            self._transform_gmm_into_diffgmm()
        elif self.gmmmode == 'intra':
            self._transform_gmm_into_intragmm()
        else:
            raise ValueError('please choose GMM mode in [None, diff, intra]')

        # estimate parameters for conversion
        self._set_Ab()
        self._set_pX()

        return

    def _set_Ab(self):
        # calculate A and b from self.jmean, self.jcov
        sddim = self.jmean.shape[1] // 2

        # calculate inverse covariance for covariance XX in each mixture
        self.covXXinv = np.zeros((self.n_mix, sddim, sddim))
        for m in range(self.n_mix):
            self.covXXinv[m] = np.linalg.inv(self.covXX[m])

        # calculate A, b, and conditional covariance given X
        self.A = np.zeros((self.n_mix, sddim, sddim))
        self.b = np.zeros((self.n_mix, sddim))
        self.cond_cov_inv = np.zeros((self.n_mix, sddim, sddim))
        for m in range(self.n_mix):
            # calculate A (i.e., A = yxcov_m * xxcov_m^-1)
            self.A[m] = self.covYX[m] @ self.covXXinv[m]

            # calculate b (i.e., b = mean^Y - A * mean^X)
            self.b[m] = self.meanY[m] - self.A[m] @ self.meanX[m]

            # calculate conditional covariance
            # (i.e., cov^(Y|X)^-1 = (yycov - A * xycov)^-1)
            self.cond_cov_inv[m] = np.linalg.inv(self.covYY[
                m] - self.A[m] @ self.covXY[m])

        return

    def _set_pX(self):
        # probability density function of X
        self.pX = sklearn.mixture.GaussianMixture(
            n_components=self.n_mix, covariance_type='full')
        self.pX.weights_ = self.w
        self.pX.means_ = self.meanX
        self.pX.covariances_ = self.covXX

        # following function is required to estimate porsterior
        self.pX.precisions_cholesky_ = _compute_precision_cholesky(
            self.covXX, 'full')
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
        self.meanX = self.meanX
        self.meanY = self.meanX
        self.covXX = self.covXX
        self.covXY = self.covXY @ np.linalg.solve(self.covYY, self.covYX)
        self.covYX = self.covXY
        self.covYY = self.covXX
        return


def get_diagonal_precision_matrix(T, D, covseq):
    return scipy.sparse.block_diag(covseq, format='csr')
