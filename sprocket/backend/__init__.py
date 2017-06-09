# coding: utf-8

import yaml
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
    - Unsupported SPTK
    - hard writing to save some acoustic features (save_hdf5)
    - write more attributes

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

    def __init__(self, yml):
        with open(yml) as yf:
            conf = yaml.safe_load(yf)

        # read parameters from yml file
        self.fs = conf['wav']['fs']
        self.shiftms = conf['wav']['shiftms']
        self.minf0 = conf['f0']['minf0']
        self.maxf0 = conf['f0']['maxf0']
        self.dim = conf['mcep']['dim']
        self.alpha = conf['mcep']['alpha']
        self.analyzer = conf['analyzer']

        # analyzer setting
        if self.analyzer == 'world':
            self.analyzer = WORLD(
                period=self.shiftms, fs=self.fs, f0_floor=self.minf0, f0_ceil=self.maxf0)
        elif self.analyzer == 'SPTK':
            raise('SPTK does not support yet, please use "world" instead.')
        else:
            raise(
                'other analyzer does not support, please use "world" instead')

    def set_wavf(self, wavf):
        # read wav file
        _, x = wavfile.read(wavf)
        self.x = np.array(x, dtype=np.float)

        return

    def analyze_all(self):
        # analysis
        self.features = self.analyzer.analyze(self.x)
        self.mcep = spgram2mcgram(
            self.features.spectrum_envelope, self.dim, self.alpha)
        self.npow = spgram2npow(self.features.spectrum_envelope)
        return

    def analyze_f0(self):
        return self.analyzer.analyze_f0(self.x).f0

    def analyze_mcep(self):
        features = self.analyzer.analyze(self.x)
        return spgram2mcgram(features.spectrum_envelope, self.dim, self.alpha)

    def save_hdf5(self, wavf):
        h5f = wavf.replace('wav', 'h5')
        hdf = HDF5(h5f, mode="w")
        hdf.save(self.features.f0, ext="f0")
        hdf.save(self.features.spectrum_envelope, ext="spc")
        hdf.save(self.features.aperiodicity, ext="ap")
        hdf.save(self.mcep, ext="mcep")
        hdf.close()
        return

    def read_hdf5(self, wavf):
        h5f = wavf.replace('wav', 'h5')
        hdf = HDF5(h5f, mode="r")
        spc = hdf.read(ext='spc')
        hdf.close()

        if self.features.spectrum_envelope[0,0] == spc[0,0]:
            print ("True")
        else:
            print ("False")
            print (self.features.spectrum_envelope[0,0], spc[0,0])

        return
