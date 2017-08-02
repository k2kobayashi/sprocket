from __future__ import division, print_function, absolute_import

import unittest

import numpy as np
from sprocket.util.delta import delta, construct_static_and_delta_matrix


class DeltaFunctionsTest(unittest.TestCase):

    def test_delta(self):
        data1d = np.random.rand(100)
        delta1d = delta(data1d)
        assert len(data1d) == len(delta1d)

        data2d = data1d.reshape(50, 2)
        delta2d = delta(data2d)
        assert delta2d.shape[0] == data2d.shape[0]
        assert delta2d.shape[1] == data2d.shape[1]

    def test_construct_W_matrix(self):
        T, D = 100, 4
        W = construct_static_and_delta_matrix(T, D)
        assert W.shape[0] == 2 * T * D
        assert W.shape[1] == T * D
