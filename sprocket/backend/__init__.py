# coding: utf-8

import numpy as np
from scipy.io import wavfile

from sprocket.backend.analyzer import WORLD
from sprocket.backend.parameterizer import spgram2mcgram
from sprocket.backend.npow import spgram2npow
from sprocket.util.hdf5 import HDF5


class FeatureExtractor(object):

    """
    analysis and save several types of acoustic feature corresponding to a single wave file
    - supported acoustic feature: f0, spc, ap, mcep, npow
    - future support?: LPC, cep, MFCC

    known issue:
    - write all attributes

    Attributes
    ----------
    fs: int
      sampling frequency
    shiftm: int
      shift size for STFT
    minf0 : float
      floor in f0 estimation
    maxf0: float
      ceil in f0 estimation
    dim: int
      order of mel-cepstrum
    alpha: int
      parameter for all-path filter to extract mel-cepstrum
    analyzer: str
      kinds of speech feature analyzer
    """

    def __init__(self, conf):
        # read parameters from speaker configure
        self.fs = conf.fs
        self.shiftms = conf.shiftms
        self.minf0 = conf.minf0
        self.maxf0 = conf.maxf0
        self.dim = conf.dim
        self.alpha = conf.alpha
        self.analyzer = conf.analyzer

        # analyzer setting
        if self.analyzer == 'world':
            self.analyzer = WORLD(period=self.shiftms,
                                  fs=self.fs,
                                  f0_floor=self.minf0,
                                  f0_ceil=self.maxf0)
        else:
            raise('No other analyzer supports, please use "world" instead')

    def set_wavf(self, wavf):
        # read wav file
        wavfs, x = wavfile.read(wavf)
        self.x = np.array(x, dtype=np.float)
        self.waveform_length = len(x)
        assert self.fs == wavfs

        return

    def analyze_all(self):
        # analysis
        self.f0, self.spc, self.ap = self.analyzer.analyze(self.x)
        self.mcep = spgram2mcgram(self.spc, self.dim, self.alpha)
        self.npow = spgram2npow(self.spc)
        return

    def analyze_f0(self):
        return self.analyzer.analyze_f0(self.x).f0

    def analyze_mcep(self):
        _, spc, _ = self.analyzer.analyze(self.x)
        return spgram2mcgram(spc, self.dim, self.alpha)

    def save_hdf5(self, wavf):
        h5f = wavf.replace('wav', 'h5')
        hdf = HDF5(h5f, mode="w")
        hdf.save(self.f0, ext="f0")
        hdf.save(self.spc, ext="spc")
        hdf.save(self.ap, ext="ap")
        hdf.save(self.mcep, ext="mcep")
        hdf.save(self.npow, ext="npow")
        hdf.save(self.waveform_length, ext="waveform_length")
        hdf.close()
        return

    def read_hdf5(self, wavf):
        h5f = wavf.replace('wav', 'h5')
        hdf = HDF5(h5f, mode="r")
        hdf.close()

        return
