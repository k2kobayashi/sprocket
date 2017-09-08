# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import resample, firwin, lfilter
from scipy.interpolate import interp1d

from .wsola import WSOLA
from ..feature import FeatureExtractor
from ..feature.synthesizer import Synthesizer


class Shifter:

    """Shifter class

    This class offers to transform f0 of input waveform
    based on WSOLA and resampling

    Parameters
    ----------
    fs : int
        Sampling frequency

    speech_rate : float
        Relative speech rate of duration modification speech to original speech

    frame_ms : int, optional
        length of frame

    completion : bool, optional
        Completion of high frequency range of F0 transformed wavform based on
        unvoiced analysis/synthesis voice of given voice and high-pass filter.
        This is due to loose the high frequency range caused by resampling
        when F0ratio setting to smaller than 1.0.

    Attributes
    ----------
    win : array
        Window vector

    """

    def __init__(self, fs, f0rate, frame_ms=20, completion=False):
        self.fs = fs
        self.f0rate = f0rate

        self.frame_ms = frame_ms  # frame length [ms]
        self.shift_ms = frame_ms // 2  # shift size for over-lap add
        self.sl = int(self.fs * self.shift_ms / 1000)  # of samples in a shift
        self.fl = int(self.fs * self.frame_ms / 1000)  # of samples in a frame
        self.epstep = int(self.sl / self.f0rate)  # step size for WSOLA
        self.win = np.hanning(self.fl)  # window function for a frame

        self.wsola = WSOLA(fs, 1 / f0rate,
                           frame_ms=self.frame_ms, shift_ms=self.shift_ms)
        self.completion = completion

    def f0transform(self, data):
        """Transform F0 of given waveform signals using

        Parameters
        ---------
        data : array, shape ('len(data)')
            array of waveform sequence

        Returns
        ---------
        transformed : array, shape (`len(data)`)
            Array of F0 transformed waveform sequence

        """

        self.xlen = len(data)

        # WSOLA
        wsolaed = self.wsola.duration_modification(data)

        # resampling
        transformed = resample(wsolaed, self.xlen)

        # Frequency completion when decrease F0 of wavform
        if self.completion:
            if self.f0rate < 1.0:
                raise ValueError("Do not enable completion if f0rate > 1.")
            transformed = self._high_frequency_completion(data, transformed)

        return transformed

    def resampling_by_interpolate(self, data):
        """Resampling base on 1st order interpolation

        Parameters
        ---------
        data : array, shape ('int(len(data) * f0rate)')
            array of wsolaed waveform

        Returns
        ---------
        wsolaed: array, shape (`len(data)`)
            Array of resampled (F0 transformed) waveform sequence

        """

        # interpolate
        wedlen = len(data)
        intpfunc = interp1d(np.arange(wedlen), data, kind=1)
        x_new = np.arange(0.0, wedlen - 1, self.f0rate)
        resampled = intpfunc(x_new)

        return resampled

    def _high_frequency_completion(self, data, transformed):
        """
        Please see Sect. 3.2 and 3.3 in the following paper to know why we complete the
        unvoiced synthesized voice of the original voice into high frequency range
        of F0 transformed voice.

        - K. Kobayashi et al., "F0 transformation techniques for statistical voice
        conversion with direct waveform modification with spectral differential,"
        Proc. IEEE SLT 2016, pp. 693-700. 2016.
        """
        # construct feature extractor and synthesis
        feat = FeatureExtractor(data, fs=self.fs)
        feat.analyze()
        uf0 = np.zeros(len(feat.f0()))

        # synthesis
        synth = Synthesizer()
        unvoice_anasyn = synth.synthesis_spc(uf0, feat.spc(),
                                             feat.ap(), fs=self.fs)

        # HPF for synthesized speech
        fil = firwin(255, self.f0rate, pass_zero=False)
        HPFed_unvoice_anasyn = lfilter(fil, 1, unvoice_anasyn)

        if HPFed_unvoice_anasyn > len(transformed):
            return transformed + HPFed_unvoice_anasyn[:len(transformed)]
        else:
            transformed[:len(HPFed_unvoice_anasyn)] += HPFed_unvoice_anasyn
            return transformed
