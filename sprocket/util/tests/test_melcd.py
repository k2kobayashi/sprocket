from __future__ import division, print_function, absolute_import

import unittest

import numpy as np
from dtw import dtw


import pysptk
from sprocket.util.distance import melcd, normalized_melcd


def get_random_peseudo_mcep(order=24, alpha=0.41):
    T, N = 10, 512
    frames = np.random.rand(T, N) * pysptk.blackman(N)
    mc = pysptk.mcep(frames, order=order, alpha=alpha)
    return mc


class DistanceTest(unittest.TestCase):

    def test_melcd(self):
        org = get_random_peseudo_mcep()
        tar = get_random_peseudo_mcep()

        # perform dtw for mel-cd function test
        def distance_func(x, y): return melcd(x, y)
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
