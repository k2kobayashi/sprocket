# -*- coding: utf-8 -*-

import numpy as np

from .analyzer import WORLD
from .parameterizer import spgram2npow

import pysptk
import pyworld


class FeatureExtractor(object):

    """Analysis and synthesize acoustic features from a given waveform

    This class allows to extract several types of acoustic features such as F0, aperiodicity,
    spectral envelope, from a given waveform.

    Parameters
    ----------
    x : array
        Vector of waveform signal

    analyzer : str, optional
        Analyzer of acoustic feature

    fs : int, optional
        Sampling frequency

    shiftms : int, optional
        Shift size for short-time Fourier transform [ms]

    minf0 : float, optional
        Floor value for F0 estimation

    maxf0 : float, optional
        Ceil value fo F0 estimation

    Attributes
    ----------
    x : array
        Vector of the given waveform signal

    f0 : array, shape (`T`,)
        Stores the analized F0 sequence

    spc : array, shape (`T`, `fftl / 2 + 1`)
        Stores the analized spectral envelope sequence

    mcep : array, shape (`T`, `dim`)
        Stores the analized mel-cepstrum sequence

    ap : array, shape (`T`, `fftl / 2 + 1`)
        Stores the analized aperiodicity sequence

    bandap : array, shape (`T`, `dim`)
        Stores the analized band-averaged aperiodicity sequence

    npow : array, shape (`T`,)
        Stores the analized normalized power sequence

    """

    def __init__(self, x, analyzer='world', fs=16000, shiftms=5, minf0=30, maxf0=700):
        self.x = np.array(x, dtype=np.float)
        self.analyzer = analyzer
        self.fs = fs
        self.shiftms = shiftms
        self.minf0 = minf0
        self.maxf0 = maxf0

        # analyzer setting
        if self.analyzer == 'world':
            self.analyzer = WORLD(period=self.shiftms,
                                  fs=self.fs,
                                  f0_floor=self.minf0,
                                  f0_ceil=self.maxf0)
        else:
            raise(
                'Other analyzer does not support, please use "world" instead')

        self._f0 = None
        self._spc = None
        self._ap = None

    def analyze(self):
        """Analyzer acoustic features using analyzer
            Following acoustic features are analized:
                world: F0, spc, ap
        """
        self._f0, self._spc, self._ap = self.analyzer.analyze(self.x)

        return

    def f0(self):
        """Return F0 sequence

        Returns
        -------
        f0 : array, shape (`T`,)
            F0 sequence

        """

        return self._f0

    def spc(self):
        """Return spectral envelope sequence

        Returns
        -------
        spc : array, shape (`T`, `fftl / 2 + 1`)
            Spectral envelope sequence

        """

        return self._spc

    def ap(self):
        """Return aperiodicity sequence

        Returns
        -------
        ap: array, shape (`T`, `fftl / 2 + 1`)
            aperiodicity sequence

        """

        return self._ap

    def mcep(self, dim=24, alpha=0.42):
        """Return mel-cepstrum sequence parameterized from spc

        Parameters
        ----------
        dim : int, optional
            Dimension of the mel-cepstrum sequence

        alpha : int, optional
            Parameter of all-path fileter for frequency transformation

        Returns
        -------
        mcep : array, shape (`T`, `dim + 1`)
            Mel-cepstrum sequence of the waveform

        """
        self._analyzed_check()

        return pysptk.sp2mc(self._spc, dim, alpha)

    def bandap(self):
        """Return band-averaged aperiodicity sequence

        Parameters
        ------
        dim: int, optional
            Dimension of the band-aperiodiciy

        Returns
        -------
        bandap: array, shape (n_frames, dim)
            Aperiodicity sequence of the waveform

        """
        self._analyzed_check()

        return pyworld.code_aperiodicity(self._ap, self.fs)

    def npow(self):
        """Return normalized power sequence calculated using analized spc

        Returns
        -------
        npow: vector, shape (`T`,)
            Normalized power sequence of the given waveform

        """
        self._analyzed_check()

        return spgram2npow(self._spc)

    def _analyzed_check(self):
        if self._f0 is None and self._spc is None and self._ap is None:
            raise(
                'Please call FeatureExtractor.analyze() before get acoustic features.')
