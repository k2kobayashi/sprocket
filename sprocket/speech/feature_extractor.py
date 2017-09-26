# -*- coding: utf-8 -*-

import pysptk
import pyworld
import numpy as np

from .analyzer import WORLD
from .parameterizer import spc2npow


class FeatureExtractor(object):

    """Analyze and synthesize acoustic features from a waveform

    Extract several types of acoustic features such as F0, aperiodicity,
    spectral envelope, from a given waveform.

    Parameters
    ----------
    analyzer : str, optional
        Analyzer of acoustic feature
        'world' : WORLD analysis/synthesis framework
        Default set to world
    fs : int, optional
        Sampling frequency of the waveform
        Default set to 16000
    fftl: int, optional
        FFT length
        Default set to 1024
    shiftms : int, optional
        Shift size for short-time Fourier transform [ms]
        Default set to 5
    minf0 : float, optional
        Floor value for F0 estimation
        Default set to 50
    maxf0 : float, optional
        Ceil value for F0 estimation
        Default set to 500

    """

    def __init__(self, analyzer='world', fs=16000, fftl=1024, shiftms=5,
                 minf0=50, maxf0=500):
        self.analyzer = analyzer
        self.fs = fs
        self.fftl = fftl
        self.shiftms = shiftms
        self.minf0 = minf0
        self.maxf0 = maxf0

        # analyzer setting
        if self.analyzer == 'world':
            self.analyzer = WORLD(fs=self.fs,
                                  fftl=self.fftl,
                                  minf0=self.minf0,
                                  maxf0=self.maxf0,
                                  shiftms=self.shiftms
                                  )
        else:
            raise(
                'Other analyzer does not support, please use "world" instead')

        self._f0 = None
        self._spc = None
        self._ap = None

    def analyze(self, x):
        """Analyze acoustic features using analyzer

        Parameters
        ----------
        x : array
            Array of waveform samples

        Returns
        -------
        f0 : array, shape (`T`,)
            F0 sequence
        spc : array, shape (`T`, `fftl / 2 + 1`)
            Spectral envelope sequence
        ap: array, shape (`T`, `fftl / 2 + 1`)
            aperiodicity sequence
        """

        self.x = np.array(x, dtype=np.float)
        self._f0, self._spc, self._ap = self.analyzer.analyze(self.x)

        # check non-negative for F0
        self._f0[self._f0 < 0] = 0

        if np.sum(self._f0) == 0.0:
            print("WARNING: F0 values are all zero.")

        return self._f0, self._spc, self._ap

    def analyze_f0(self, x):
        """Analyze F0 using analyzer

        Parameters
        ----------
        x : array
            Array of waveform samples

        Returns
        -------
        f0 : array, shape (`T`,)
            F0 sequence
        """

        self.x = np.array(x, dtype=np.float)
        self._f0 = self.analyzer.analyze_f0(self.x)

        # check non-negative for F0
        self._f0[self._f0 < 0] = 0

        if np.sum(self._f0) == 0.0:
            print("WARNING: F0 values are all zero.")

        return self._f0

    def mcep(self, dim=24, alpha=0.42):
        """Return mel-cepstrum sequence parameterized from spectral envelope

        Parameters
        ----------
        dim : int, optional
            Dimension of the mel-cepstrum sequence
        alpha : int, optional
            Parameter of all-path fileter for frequency transformation

        Returns
        -------
        mcep : array, shape (`T`, `dim + 1`)
            Mel-cepstrum sequence

        """
        self._analyzed_check()

        return pysptk.sp2mc(self._spc, dim, alpha)

    def codeap(self):
        """Return coded aperiodicity sequence

        Returns
        -------
        codeap : array, shape (`T`, `dim`)
            Encoded aperiodicity sequence of the waveform
            The `dim` of code ap is defined based on the `fs` as follow:
            fs = `16000` : `1`
            fs = `22050` : `2`
            fs = `44100` : `5`
            fs = `48000` : `5`
        """

        self._analyzed_check()

        return pyworld.code_aperiodicity(self._ap, self.fs)

    def npow(self):
        """Return normalized power sequence parameterized from spectral envelope

        Returns
        -------
        npow: vector, shape (`T`,)
            Normalized power sequence of the given waveform

        """
        self._analyzed_check()

        return spc2npow(self._spc)

    def _analyzed_check(self):
        if self._f0 is None and self._spc is None and self._ap is None:
            raise('Call FeatureExtractor.analyze() before get parameterized features.')
