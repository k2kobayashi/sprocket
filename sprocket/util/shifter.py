# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d

from .wsola import WSOLA


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

    shift_ms : int, optional
        length of shift

    Attributes
    ----------
    win : array
        Window vector

    """

    def __init__(self, fs, f0rate, frame_ms=20, shift_ms=10):
        self.fs = fs
        self.f0rate = f0rate

        self.frame_ms = frame_ms  # frame length [ms]
        self.shift_ms = shift_ms  # shift length [ms]
        self.sl = int(self.fs * self.shift_ms / 1000)  # of samples in a shift
        self.fl = int(self.fs * self.frame_ms / 1000)  # of samples in a frame
        self.epstep = int(self.sl / self.f0rate)  # step size for WSOLA
        self.win = np.hanning(self.fl)  # window function for a frame

        self.wsola = WSOLA(fs, 1 / f0rate,
                           frame_ms=self.frame_ms, shift_ms=self.shift_ms)

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
