import unittest

import os
import numpy as np
from scipy.io import wavfile
from scipy.fftpack import fft

from sprocket.speech import Shifter

dirpath = os.path.dirname(os.path.realpath(__file__))
saveflag = False


class ShifterTest(unittest.TestCase):

    def test_shifter(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        for f0rate in (0.5, 0.75, 1.0, 1.5, 2.0):
            if f0rate < 1:
                completion = True
            else:
                completion = False
            shifter = Shifter(fs, f0rate=f0rate)
            transformed_x = shifter.f0transform(x, completion=completion)
            assert len(x) == len(transformed_x)

            if saveflag:
                fpath = path + str(f0rate) + '.wav'
                wavfile.write(fpath, fs, transformed_x.astype(np.int16))

    def test_high_frequency_completion(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)

        f0rate = 0.5
        shifter = Shifter(fs, f0rate=f0rate)
        mod_x = shifter.f0transform(x, completion=False)
        mod_xc = shifter.f0transform(x, completion=True)
        assert len(mod_x) == len(mod_xc)

        N = 512
        fl = int(fs * 25 / 1000)
        win = np.hanning(fl)
        sts = [1000, 5000, 10000, 20000]
        for st in sts:
            # confirm w/o completion
            f_mod_x = fft(mod_x[st: st + fl] / 2**16 * win)
            amp_mod_x = 20.0 * np.log10(np.abs(f_mod_x))

            # confirm w/ completion
            f_mod_xc = fft(mod_xc[st: st + fl] / 2**16 * win)
            amp_mod_xc = 20.0 * np.log10(np.abs(f_mod_xc))

            assert np.mean(amp_mod_x[N // 4:] < np.mean(amp_mod_xc[N // 4:]))
