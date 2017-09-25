from __future__ import division, print_function, absolute_import

import unittest
import os

import numpy as np
from scipy.io import wavfile
import pyworld
from sprocket.speech import FeatureExtractor, Synthesizer

dirpath = os.path.dirname(os.path.realpath(__file__))
minf0 = 60


class AnalysisSynthesisTest(unittest.TestCase):

    def test_anasyn_16000(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(analyzer='world', fs=fs, shiftms=5, fftl=1024)
        f0, spc, ap = af.analyze(x)
        mcep = af.mcep(dim=24, alpha=0.42)

        assert len(np.nonzero(f0)[0]) > 0
        assert spc.shape == ap.shape

        # synthesize F0, mcep, ap
        synth = Synthesizer(fs=fs, fftl=1024, shiftms=5)
        wav = synth.synthesis(f0, mcep, ap, alpha=0.42)
        nun_check(wav)

    def test_anasyn_44100(self):
        path = dirpath + '/data/test44100.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(analyzer='world', fs=fs, shiftms=5, minf0=100, fftl=2048)
        f0, spc, ap = af.analyze(x)
        mcep = af.mcep(dim=40, alpha=0.50)

        assert len(np.nonzero(f0)[0]) > 0
        assert spc.shape == ap.shape

        # mcep synthesis
        synth = Synthesizer(fs=fs, fftl=2048, shiftms=5)
        wav = synth.synthesis(f0, mcep, ap, alpha=0.50)
        nun_check(wav)

    def test_spc_and_npow(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(analyzer='world', fs=fs, shiftms=5)
        _, spc, _ = af.analyze(x)
        npow = af.npow()
        assert spc.shape[0] == npow.shape[0]

    def test_synthesis_from_codeap(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        af = FeatureExtractor(analyzer='world', fs=fs, shiftms=5)
        f0, spc, ap = af.analyze(x)
        codeap = af.codeap()

        assert len(np.nonzero(f0)[0]) > 0
        assert spc.shape == ap.shape

        assert pyworld.get_num_aperiodicities(fs) == codeap.shape[-1]
        ap = pyworld.decode_aperiodicity(codeap, fs, 1024)

        synth = Synthesizer(fs=fs, fftl=1024, shiftms=5)
        wav = synth.synthesis_spc(f0, spc, ap)
        nun_check(wav)


def nun_check(wav):
    if any(np.isnan(wav)):
        raise ('wavform consists NaN value.')
