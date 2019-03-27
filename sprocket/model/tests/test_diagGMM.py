# -*- coding: utf-8 -*-

import unittest

import os
import numpy as np
from sprocket.model import GMMTrainer, GMMConvertor
from sprocket.util import delta

dirpath = os.path.dirname(os.path.realpath(__file__))


class ModeldiagGMMTest(unittest.TestCase):

    def test_blockdiag_GMM_train_and_convert(self):
        jnt = np.random.rand(100, 20)
        gmm_tr = GMMTrainer(n_mix=4, n_iter=100, covtype='block_diag')
        gmm_tr.train(jnt)

        gmm_cv = GMMConvertor(
            n_mix=4, covtype='block_diag', gmmmode=None)
        gmm_cv.open_from_param(gmm_tr.param)

        data = np.random.rand(200, 5)
        sddata = np.c_[data, delta(data)]
        odata = gmm_cv.convert(sddata, cvtype='mlpg')
        odata = gmm_cv.convert(sddata, cvtype='mmse')
        assert data.shape == odata.shape

        # test for singlepath
        Ajnt = np.random.rand(100, 120)
        Bjnt = np.random.rand(100, 140)
        gmm_tr.estimate_responsibility(jnt)
        Aparam = gmm_tr.train_singlepath(Ajnt)
        Bparam = gmm_tr.train_singlepath(Bjnt)
        assert np.allclose(Aparam.weights_, Bparam.weights_)
