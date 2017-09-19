from __future__ import division, print_function, absolute_import

import unittest
import os

from scipy.io import wavfile

from sprocket.speech import FeatureExtractor, mod_power

dirpath = os.path.dirname(os.path.realpath(__file__))


class ModifyPowerTest(unittest.TestCase):

    def test_mod_power(self):
        path = os.path.join(dirpath, 'data', 'test16000.wav')
        fs, x = wavfile.read(path)
        af = FeatureExtractor(analyzer='world', fs=fs, shiftms=5)
        f0, _, ap = af.analyze(x)
        mcep = af.mcep(dim=24, alpha=0.42)

        rmcep = mcep
        cvmcep = mcep * 1.50

        modified_cvmcep = mod_power(cvmcep, rmcep, alpha=0.42)

        assert modified_cvmcep.shape == cvmcep.shape
