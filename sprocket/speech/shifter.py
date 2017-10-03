# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import resample, firwin, lfilter
from scipy.interpolate import interp1d

from .wsola import WSOLA
from .feature_extractor import FeatureExtractor
from .synthesizer import Synthesizer


class Shifter(object):

    """Shifter class

    Transform f0 of given waveform signal based on WSOLA and
    resampling

    Parameters
    ----------
    fs : int
        Sampling frequency
    f0rate: float
        F0 transformation ratio
    shiftms : int, optional
        length of shift size [ms]

    Attributes
    ----------
    win : array
        Window vector

    """

    def __init__(self, fs, f0rate, shiftms=10):
        self.fs = fs
        self.f0rate = f0rate

        self.shiftms = shiftms  # shift size for over-lap add
        self.wsola = WSOLA(fs, 1 / f0rate, shiftms=self.shiftms)

    def f0transform(self, x, completion=False):
        """Transform F0 of given waveform signals using

        Parameters
        ---------
        x : array, shape ('len(x)')
            array of waveform sequence

        completion : bool, optional
        Completion of high frequency range of F0 transformed wavform based on
        unvoiced analysis/synthesis voice of given voice and high-pass filter.
        This is due to loose the high frequency range caused by resampling
        when F0ratio setting to smaller than 1.0.

        Returns
        ---------
        transformed : array, shape (`len(x)`)
            Array of F0 transformed waveform sequence

        """

        self.xlen = len(x)

        # WSOLA
        wsolaed = self.wsola.duration_modification(x)

        # resampling
        transformed = resample(wsolaed, self.xlen)

        # Frequency completion when decrease F0 of wavform
        if completion:
            if self.f0rate > 1.0:
                raise ValueError("Do not enable completion if f0rate > 1.")
            transformed = self._high_frequency_completion(x, transformed)

        return transformed

    def resampling_by_interpolate(self, x):
        """Resampling base on 1st order interpolation

        Parameters
        ---------
        x : array, shape ('int(len(x) * f0rate)')
            array of wsolaed waveform

        Returns
        ---------
        wsolaed: array, shape (`len(x)`)
            Array of resampled (F0 transformed) waveform sequence

        """

        # interpolate
        wedlen = len(x)
        intpfunc = interp1d(np.arange(wedlen), x, kind=1)
        x_new = np.arange(0.0, wedlen - 1, self.f0rate)
        resampled = intpfunc(x_new)

        return resampled

    def _high_frequency_completion(self, x, transformed):
        """
        Please see Sect. 3.2 and 3.3 in the following paper to know why we complete the
        unvoiced synthesized voice of the original voice into high frequency range
        of F0 transformed voice.

        - K. Kobayashi et al., "F0 transformation techniques for statistical voice
        conversion with direct waveform modification with spectral differential,"
        Proc. IEEE SLT 2016, pp. 693-700. 2016.
        """
        # construct feature extractor and synthesis
        feat = FeatureExtractor(fs=self.fs)
        f0, spc, ap = feat.analyze(x)
        uf0 = np.zeros(len(f0))

        # synthesis
        synth = Synthesizer(fs=self.fs)
        unvoice_anasyn = synth.synthesis_spc(uf0, spc, ap)

        # HPF for synthesized speech
        fil = firwin(255, self.f0rate, pass_zero=False)
        HPFed_unvoice_anasyn = lfilter(fil, 1, unvoice_anasyn)

        if len(HPFed_unvoice_anasyn) > len(transformed):
            return transformed + HPFed_unvoice_anasyn[:len(transformed)]
        else:
            transformed[:len(HPFed_unvoice_anasyn)] += HPFed_unvoice_anasyn
            return transformed
