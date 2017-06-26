import unittest

import os
import numpy as np
from dtw import dtw

from sprocket.util.distance import melcd, normalized_melcd

dirpath = os.path.dirname(os.path.realpath(__file__))
orgf = dirpath + '/data/test_org.mcep'
tarf = dirpath + '/data/test_tar.mcep'


class DistanceTest(unittest.TestCase):

    def test_melcd(self):
        # read mcep from file
        org = np.fromfile(orgf)
        org = org.reshape(len(org) / 25, 25)
        tar = np.fromfile(tarf)
        tar = tar.reshape(len(tar) / 25, 25)

        # perform dtw for mel-cd function test
        distance_func = lambda x, y: melcd(x, y)
        dist, cost, acost, twf = dtw(org, tar, dist=distance_func)

        # align org and tar
        orgmod = org[twf[0]]
        tarmod = tar[twf[1]]
        assert orgmod.shape == tarmod.shape

        # test for mel-cd calculation
        flen = len(twf[0])
        mcd = 0
        for t in range(flen):
            mcd += melcd(orgmod[t], tarmod[t])
        norm_mcd1 = mcd / flen
        norm_mcd2 = normalized_melcd(orgmod, tarmod)

        assert norm_mcd1 - norm_mcd2 < np.exp(-10)
