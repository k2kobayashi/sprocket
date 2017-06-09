# coding: utf-8

import os
import yaml
import h5py
import numpy as np
from scipy.io import wavfile

from sprocket.backend.world import WORLD
from sprocket.backend.npow import spgram2npow
from sprocket.backend.parameterizer import spgram2mcgram
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
        dirname, _ = os.path.split(h5f)

        # check directory
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # check existing h5 file
        if os.path.exists(h5f):
            print("overwrite because HDF5 file already exists. ")

        # write hdf5 file format
        h5 = h5py.File(h5f, "w")
        h5.create_group(dirname)
        h5.create_dataset(
            dirname + '/spc', data=self.features.spectrum_envelope)
        h5.create_dataset(dirname + '/ap', data=self.features.aperiodicity)
        h5.create_dataset(dirname + '/f0', data=self.features.f0)
        h5.create_dataset(dirname + '/mcep', data=self.mcep)
        h5.create_dataset(dirname + '/npow', data=self.npow)
        h5.flush()
        h5.close()

        return
