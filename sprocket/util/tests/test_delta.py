import unittest
import numpy as np

from sprocket.util.delta import delta

class DeltaTest(unittest.TestCase):

    def test_delta(self):
        T = 100
        dim = 2
        data_1d = np.arange(T * dim)
        delta_1d = delta(data_1d)

        data_2d = data_1d.reshape(T, dim)
        delta_12 = delta(data_2d)
