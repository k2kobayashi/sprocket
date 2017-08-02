import unittest

import os
import numpy as np
from sprocket.model.GMM import GMMTrainer, GMMConvertor
from sprocket.util.delta import delta

dirpath = os.path.dirname(os.path.realpath(__file__))
tmppath = dirpath + '/test_tmp_gmm.pkl'


class ModelGMMTest(unittest.TestCase):

    def test_GMM_train_and_convert(self):
        jnt = np.random.rand(100, 20)
        gmm_tr = GMMTrainer(n_mix=4, n_iter=100, covtype='full')
        gmm_tr.train(jnt)

        gmm_tr.save(tmppath)
        gmm_tr.open(tmppath)

        data = np.random.rand(200, 5)
        sddata = np.c_[data, delta(data)]
        gmm_cv = GMMConvertor(
            n_mix=4, covtype='full', gmmmode=None, cvtype='mlpg')
        gmm_cv.open(tmppath)
        os.remove(tmppath)

        odata = gmm_cv.convert(sddata)

        assert data.shape == odata.shape
