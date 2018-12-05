import unittest

import numpy as np
from sprocket.model import MS


class MSTest(unittest.TestCase):

    def test_MSstatistics(self):
        ms = MS()
        datalist = []
        for i in range(1, 4):
            datalist.append(np.random.rand(100 * i).reshape(100 * i // 4, 4))
        msstats = ms.estimate(datalist)

        print(msstats.shape)

        data = np.random.rand(100 * 5).reshape(100 * 5 // 4, 4)
        odata = ms.postfilter(data, msstats, msstats, startdim=0)
        assert data.shape[0] == odata.shape[0]

        odata = ms.postfilter(data, msstats, msstats, alpha=0.0, startdim=0, k=0.95)
        assert np.all(data == odata)
