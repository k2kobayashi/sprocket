import unittest

import os
import numpy as np
from sprocket.stats.gv import GV

dirpath = os.path.dirname(os.path.realpath(__file__))
fpath = dirpath + '/test_tar.gvstats'
tmppath = dirpath + '/test_tmp.gvstats'


class GVstatisticsTest(unittest.TestCase):

    def test_GV_postfilter(self):
        gvstats = GV()
        gvstats.open_from_file(fpath)

        data = np.random.rand(100 * 5).reshape(100 * 5 / 2, 2)
        odata = gvstats.postfilter(data, startdim=0)

        assert data.shape == odata.shape

    def test_estimate_GVstatistics(self):
        gvstats = GV()
        datalist = []
        for i in range(1, 4):
            datalist.append(np.random.rand(100 * i).reshape(100 * i / 2, 2))

        gvstats.estimate(datalist)
        gvstats.save(tmppath)
        os.remove(tmppath)
