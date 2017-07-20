import unittest
import os

import numpy as np
from scipy.io import wavfile

from sprocket.feature import FeatureExtractor
from sprocket.feature.synthesizer import Synthesizer

dirpath = os.path.dirname(os.path.realpath(__file__))


class AnalysisSynthesisTest(unittest.TestCase):

    def test_anasyn_16000(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(x, analyzer='world', fs=fs, shiftms=5)
        af.analyze()
        f0 = af.f0
        mcep = af.mcep(dim=24, alpha=0.42)
        ap = af.ap
        synth = Synthesizer()

        # mcep synthesis
        wav = synth.synthesis(
            f0, mcep, ap, alpha=0.42, fftl=1024, fs=fs, shiftms=5)
        nun_check(wav)
        opath = dirpath + '/data/anasyn16000.wav'
        wavfile.write(opath, fs, np.array(wav, dtype=np.int16))

    def test_anasyn_44100(self):
        path = dirpath + '/data/test44100.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(x, analyzer='world', fs=fs, shiftms=5)
        af.analyze()
        f0 = af.f0
        mcep = af.mcep(dim=40, alpha=0.50)
        ap = af.ap
        synth = Synthesizer()

        # mcep synthesis
        wav = synth.synthesis(
            f0, mcep, ap, alpha=0.50, fftl=2048, fs=fs, shiftms=5)
        nun_check(wav)
        opath = dirpath + '/data/anasyn44100.wav'
        wavfile.write(opath, fs, np.array(wav, dtype=np.int16))

    def test_spc_and_npow(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(x, analyzer='world', fs=fs, shiftms=5)
        af.analyze()
        spc = af.spc
        npow = af.npow()
        assert spc.shape[0] == npow.shape[0]


def nun_check(wav):
    if any(np.isnan(wav)):
        raise ('wavform consists NaN value.')
