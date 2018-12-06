import unittest

import numpy as np
from sprocket.model import GMMTrainer, GMMConvertor
from sprocket.util import static_delta


class BDGMMTest(unittest.TestCase):

    def test_BlockDiagonalGMM(self):
        jnt = np.random.rand(1000, 20)
        gmm_tr = GMMTrainer(n_mix=32, n_iter=100, covtype='block_diag')
        gmm_tr.train(jnt)

        data = np.random.rand(200, 5)
        sddata = static_delta(data)
        gmm_cv = GMMConvertor(
            n_mix=32, covtype='full', gmmmode=None)
        gmm_cv.open_from_param(gmm_tr.param)

        odata = gmm_cv.convert(sddata, cvtype='mlpg')
        odata = gmm_cv.convert(sddata, cvtype='mmse')

        assert data.shape == odata.shape
