import unittest

import numpy as np
from sprocket.model import F0statistics


class F0statisticsTest(unittest.TestCase):

    def test_estimate_F0statistics(self):
        f0stats = F0statistics()
        orgf0s = []
        for i in range(1, 4):
            orgf0s.append(200 * np.r_[np.random.rand(100 * i), np.zeros(100)])
        orgf0stats = f0stats.estimate(orgf0s)

        tarf0s = []
        for i in range(1, 8):
            tarf0s.append(300 * np.r_[np.random.rand(100 * i), np.zeros(100)])
        tarf0stats = f0stats.estimate(tarf0s)

        orgf0 = 200 * np.r_[np.random.rand(100 * i), np.zeros(100)]
        cvf0 = f0stats.convert(orgf0, orgf0stats, tarf0stats)

        assert len(orgf0) == len(cvf0)
