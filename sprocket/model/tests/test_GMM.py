import unittest

import numpy as np
from sprocket.model import GMMTrainer, GMMConvertor
from sprocket.util import delta


class ModelGMMTest(unittest.TestCase):

    def test_GMM_train_and_convert(self):
        jnt = np.random.rand(100, 20)
        gmm_tr = GMMTrainer(n_mix=4, n_iter=100, covtype='full')
        gmm_tr.train(jnt)

        data = np.random.rand(200, 5)
        sddata = np.c_[data, delta(data)]
        gmm_cv = GMMConvertor(
            n_mix=4, covtype='full', gmmmode=None)
        gmm_cv.open_from_param(gmm_tr.param)

        odata = gmm_cv.convert(sddata, cvtype='mlpg')
        odata = gmm_cv.convert(sddata, cvtype='mmse')

        assert data.shape == odata.shape
