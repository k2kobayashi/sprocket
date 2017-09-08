# -*- coding: utf-8 -*-

import numpy as np

from scipy.signal import correlate2d
from skimage.util import view_as_windows


class WSOLA:

    """WSOLA class

    This class offers to modify speech duration

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

    def __init__(self, fs, speech_rate, frame_ms=20, shift_ms=10):
        self.fs = fs
        self.speech_rate = speech_rate

        self.frame_ms = frame_ms  # frame length [ms]
        self.shift_ms = shift_ms  # shift length [ms]
        self.sl = int(self.fs * self.shift_ms / 1000)  # of samples in a shift
        self.fl = int(self.fs * self.frame_ms / 1000)  # of samples in a frame
        self.epstep = int(self.sl * self.speech_rate)  # step size for WSOLA
        self.win = np.hanning(self.fl)  # window function for a frame

    def duration_modification(self, data):
        """Duration modification based on WSOLA

        Parameters
        ---------
        data : array, shape ('len(data)')
            array of waveform sequence

        Returns
        ---------
        wsolaed: array, shape (`int(len(data) / speech_rate)`)
            Array of WSOLAed waveform sequence

        """

        wlen = len(data)
        wsolaed = np.zeros(int(wlen / self.speech_rate), dtype='d')

        # initialization
        sp = self.sl
        rp = sp + self.sl
        ep = sp + self.epstep
        outp = 0

        while wlen > ep + self.fl:
            if ep - self.fl < self.sl:
                sp += self.epstep
                rp = sp + self.sl
                ep += self.epstep
                continue

            # copy wavform
            ref = data[rp - self.sl:rp + self.sl]
            buff = data[ep - self.fl:ep + self.fl]

            # search minimum distance bepween ref and buff
            delta = self._search_minimum_distance(ref, buff)
            epd = ep + delta

            # store WSOLAed waveform using over-lap add
            spdata = data[sp:sp + self.sl] * self.win[self.sl:]
            epdata = data[epd - self.sl: epd] * self.win[:self.sl]
            wsolaed[outp:outp + self.sl] = spdata + epdata
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
