import unittest

import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from sprocket.util.filter import low_pass_filter, high_pass_filter

saveflag = False


class FilterFunctionsTest(unittest.TestCase):

    def test_filter(self):
        fs = 8000
        f0 = 440
        sin = np.array([np.sin(2.0 * np.pi * f0 * n / fs)
                        for n in range(fs * 1)])
        noise = np.random.rand(len(sin)) - 0.5
        wav = sin + noise
        lpfed = low_pass_filter(wav, 500, n_taps=255, fs=fs)
        hpfed = high_pass_filter(wav, 1000, n_taps=255, fs=fs)

        lpfed_2d = low_pass_filter(np.vstack([wav, noise]).T, 500, fs=fs)
        hpfed_2d = high_pass_filter(np.vstack([wav, noise]).T, 1000, fs=fs)

        if saveflag:
            plt.figure()
            plt.plot(lpfed, label='lpf')
            plt.plot(hpfed, label='hpf')
            plt.legend()
            plt.xlim(0, 100)
            plt.savefig('filter.png')
