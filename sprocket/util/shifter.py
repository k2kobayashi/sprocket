# -*- coding: utf-8 -*-

import skimage
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d


class Shifter:

    """Shifter class
    This class offers to transform f0 of input waveform
    based on WSOLA and resampling

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

    def transform(self, data):
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
        wsolaed = self.duration_modification(data)

        # resampling
        transformed = self.resampling(wsolaed)

        return transformed

    def duration_modification(self, data):
        """Duration modification based on WSOLA

        Parameters
        ---------
        data : array, shape ('len(data)')
            array of waveform sequence

        Returns
        ---------
        wsolaed: array, shape (`int(len(data) * f0rate)`)
            Array of WSOLAed waveform sequence

        """

        wlen = len(data)
        wsolaed = np.zeros(int(wlen * self.f0rate), dtype='d')

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

    def resampling(self, data):
        """Resampling

        Parameters
        ---------
        data : array, shape ('int(len(data) * f0rate)')
            array of wsolaed waveform

        Returns
        ---------
        resampled : array, shape (`len(data)`)
            Array of resampled (F0 transformed) waveform sequence

        """

        return scipy.signal.resample(data, self.xlen)

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

    def _search_minimum_distance(self, ref, buff):
        if len(ref) < self.fl:
            ref = np.r_[ref, np.zeros(self.fl - len(ref))]

        # slicing and windowing one sample by one
        buffmat = skimage.util.view_as_windows(buff, self.fl) * self.win
        refwin = np.array(ref * self.win).reshape(1, self.fl)
        corr = scipy.signal.correlate2d(buffmat, refwin, mode='valid')

        return np.argmax(corr) - self.sl

    def _cross_correration(self, org, tar):
        return np.correlate(org, tar)
