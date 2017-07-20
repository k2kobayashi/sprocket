import unittest

import os
import numpy as np
from sprocket.stats.f0statistics import F0statistics

dirpath = os.path.dirname(os.path.realpath(__file__))
orgpath = dirpath + '/test_org.f0stats'
tarpath = dirpath + '/test_tar.f0stats'
tmppath = dirpath + '/test_tmp.f0stats'


class F0statisticsTest(unittest.TestCase):

    def test_f0_convert(self):
        f0stats = F0statistics()
        f0stats.open_from_file(orgpath, tarpath)
        f0 = np.r_[200 * np.random.rand(50), np.zeros(50)]
        cvf0 = f0stats.convert(f0)
        assert len(f0) == len(cvf0)

    def test_estimate_F0statistics(self):
        f0stats = F0statistics()
        f0s = []
        for i in range(1, 4):
            f0s.append(200 * np.r_[np.random.rand(100 * i), np.zeros(100)])

        f0stats.estimate(f0s)
        f0stats.save(tmppath)
        f0stats.read(f0stats.f0stats, f0stats.f0stats)
        os.remove(tmppath)
