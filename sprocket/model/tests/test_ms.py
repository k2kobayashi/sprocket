import unittest

import numpy as np
from sprocket.model import MS
from sprocket.util import low_pass_filter

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

saveflag = True

dim = 4


class MSTest(unittest.TestCase):

    def test_MSstatistics(self):
        ms = MS()
        datalist = []
        for i in range(1, 4):
            T = 200 * i
            data = low_pass_filter(np.random.rand(T * dim).reshape(T, dim), 50, fs=200, n_taps=63)
            datalist.append(data)
        msstats = ms.estimate(datalist)

        data = np.random.rand(500 * dim).reshape(500, dim)
        data_lpf = low_pass_filter(data, 50, fs=200, n_taps=63)
        data_ms = ms.logpowerspec(data)
        data_lpf_ms = ms.logpowerspec(data_lpf)

        odata = ms.postfilter(data, msstats, msstats, startdim=0)
        odata_lpf = ms.postfilter(data_lpf, msstats, msstats, startdim=0)
        assert data.shape[0] == odata.shape[0]

        if saveflag:
            # plot sequence
            plt.figure()
            plt.plot(data[:, 0], label='data')
            plt.plot(data_lpf[:, 0], label='data_lpf')
            plt.plot(odata[:, 0], label='odata')
            plt.plot(odata_lpf[:, 0], label='odata_lpf')
            plt.xlim(0, 100)
            plt.legend()
            plt.savefig('ms_seq.png')

            # plot MS
            plt.figure()
            plt.plot(msstats[:, 0], label='msstats')
            plt.plot(data_ms[:, 0], label='data')
            plt.plot(data_lpf_ms[:, 0], label='data_lpf')
            plt.plot(ms.logpowerspec(odata)[:, 0], label='mspf data')
            plt.plot(ms.logpowerspec(odata_lpf)[:, 0], label='mspf data_lpf')
            plt.xlim(0, msstats.shape[0] // 2 + 1)
            plt.legend()
            plt.savefig('ms.png')
