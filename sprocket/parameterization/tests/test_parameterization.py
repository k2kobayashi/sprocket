import numpy as np
import os
from world import pyDioOption, dio, stonemask, cheaptrick
from sprocket.parameterization import sp2mc, mc2sp, spgram2mcgram, mcgram2spgram

import unittest

from scipy.io import wavfile

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'test16k.wav')
print path
fs, x = wavfile.read(path)
x = np.array(x, dtype=np.float)
assert fs == 16000
period = 5.0
opt = pyDioOption(40.0, 700, 2.0, period, 4)
f0, time_axis = dio(x, fs, period, opt)
f0 = stonemask(x, fs, period, time_axis, f0)
spectrogram = cheaptrick(x, fs, period, time_axis, f0)


class SpectrumEnevelopeParameterizationTest(unittest.TestCase):

    def test_sp2mc(self):
        spec = spectrogram[29]
        mc = sp2mc(spec, 25, 0.41)
        assert not np.isnan(mc).all()
        assert len(mc) == 26

    def test_spgram2mcgram(self):
        mcgram = spgram2mcgram(spectrogram, 25, 0.41)
        assert not np.isnan(mcgram).all()
        assert mcgram.shape == (spectrogram.shape[0], 26)

    def test_mc2sp(self):
        order = 25
        mc = np.ones(order + 1)
        spec = mc2sp(mc, 0.41, 1024)
        assert len(spec) == 513
        assert not np.isnan(spec).all()

    def test_spgram2mcgram(self):
        X = mcgram2spgram(spgram2mcgram(spectrogram, 25, 0.41), 0.41, 1024)
        assert not np.isnan(X).all()
        assert X.shape == spectrogram.shape

    def test_approximate_spec(self):
        spec = spectrogram[29]
        fftlen = (len(spec) - 1) * 2

        approximate_spec = mc2sp(sp2mc(spec, 25, 0.41), 0.41, fftlen)
        nmse25 = np.linalg.norm(
            np.log(spec) - np.log(approximate_spec)) / np.linalg.norm(np.log(spec))
        assert spec.shape == approximate_spec.shape
        assert nmse25 < 0.07

        approximate_spec = mc2sp(sp2mc(spec, 30, 0.41), 0.41, fftlen)
        nmse30 = np.linalg.norm(
            np.log(spec) - np.log(approximate_spec)) / np.linalg.norm(np.log(spec))
        assert nmse30 < 0.06

        approximate_spec = mc2sp(sp2mc(spec, 40, 0.41), 0.41, fftlen)
        nmse40 = np.linalg.norm(
            np.log(spec) - np.log(approximate_spec)) / np.linalg.norm(np.log(spec))
        assert nmse40 < 0.04

        assert nmse25 > nmse30 > nmse40
