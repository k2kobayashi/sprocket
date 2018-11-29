# -*- coding: utf-8 -*-

import numpy as np

from scipy.signal import correlate2d
from skimage.util import view_as_windows


class WSOLA(object):

    """WSOLA class

    Modify speech rate of a given waveform

    Parameters
    ----------
    fs : int
        Sampling frequency
    speech_rate : float
        Relative speech rate of duration modified speech to original speech
    shiftms : int, optional
        length of shift

    Attributes
    ----------
    win : array
        Window vector

    """

    def __init__(self, fs, speech_rate, shiftms=10):
        self.fs = fs
        self.speech_rate = speech_rate

        self.shiftms = shiftms  # shift length [ms]
        self.sl = int(self.fs * self.shiftms / 1000)  # of samples in a shift
        self.fl = self.sl * 2  # of samples in a frame
        self.epstep = int(self.sl * self.speech_rate)  # step size for WSOLA
        self.win = np.hanning(self.fl)  # window function for a frame

    def duration_modification(self, x):
        """Duration modification based on WSOLA

        Parameters
        ---------
        x : array, shape ('len(x)')
            array of waveform sequence

        Returns
        ---------
        wsolaed: array, shape (`int(len(x) / speech_rate)`)
            Array of WSOLAed waveform sequence

        """

        wlen = len(x)
        wsolaed = np.zeros(int(wlen / self.speech_rate), dtype='d')

        # initialization
        sp = self.sl * 2
        rp = sp + self.sl
        ep = sp + self.epstep
        outp = sp

        # allocate first frame of waveform to outp
        wsolaed[:outp] = x[:outp]

        while wlen > ep + self.fl:
            # copy wavform
            ref = x[rp - self.sl:rp + self.sl]
            buff = x[ep - self.fl:ep + self.fl]

            # search minimum distance bepween ref and buff
            delta = self._search_minimum_distance(ref, buff)
            epd = ep + delta

            # store WSOLAed waveform using over-lap add
            spdata = x[sp:sp + self.sl] * self.win[self.sl:]
            epdata = x[epd - self.sl:epd] * self.win[:self.sl]
            if len(spdata) == len(wsolaed[outp:outp + self.sl]):
                wsolaed[outp:outp + self.sl] = spdata + epdata
            else:
                wsolaed_len = len(wsolaed[outp:outp + self.sl])
                wsolaed[outp:outp + self.sl] = spdata[:wsolaed_len] + \
                    epdata[:wsolaed_len]

            outp += self.sl

            # transtion to next frame
            sp = epd
            rp = sp + self.sl
            ep += self.epstep

        return wsolaed

    def _search_minimum_distance(self, ref, buff):
        if len(ref) < self.fl:
            ref = np.r_[ref, np.zeros(self.fl - len(ref))]

        # slicing and windowing one sample by one
        buffmat = view_as_windows(buff, self.fl) * self.win
        refwin = np.array(ref * self.win).reshape(1, self.fl)
        corr = correlate2d(buffmat, refwin, mode='valid')

        return np.argmax(corr) - self.sl
