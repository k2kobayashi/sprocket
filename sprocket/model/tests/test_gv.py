import unittest

import numpy as np
from sprocket.model import GV


class GVTest(unittest.TestCase):

    def test_GVstatistics(self):
        gvstats = GV()
        datalist = []
        for i in range(1, 4):
            datalist.append(np.random.rand(100 * i).reshape(100 * i // 2, 2))
        gv = gvstats.estimate(datalist)

        data = np.random.rand(100 * 5).reshape(100 * 5 // 2, 2)
        odata = gvstats.postfilter(data, gv, startdim=0)
        assert data.shape[0] == odata.shape[0]

        odata = gvstats.postfilter(data, gv, alpha=0.0, startdim=0)
        assert np.all(data == odata)
