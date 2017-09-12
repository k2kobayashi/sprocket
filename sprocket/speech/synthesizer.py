# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import pyworld
import pysptk


class Synthesizer(object):

    """
    Speech synthesizer from several types of acoustic features

    """

    def __init__(self):
        return

    def synthesis(self, f0, mcep, ap, alpha=0.42, fftl=1024, fs=16000, shiftms=5):
        """synthesis generates waveform from F0, mcep, aperiodicity

        Parameters
        ----------
        f0: array, shape (T, `1`)
          array of F0 sequence
        mcep: array, shape (T, `self.conf.dim`)
          array of mel-cepstrum sequence
        aperiodicity: array, shape (T, `fftlen / 2 + 1`)
          array of aperiodicity
        dim: int, optional
          Dimension of the mel-cepstrum sequence
        alpha: int, optional
          Parameter of all-path fileter for frequency transformation
        fs: int, optional
          Sampling frequency
        shiftms: int, optional
          Shift size for STFT

        Return
        ------
        wav: vector
          Synethesized waveform

        """

        # mcep into spc
        spc = pysptk.mc2sp(mcep, alpha, fftl)

        # generate waveform using world vocoder with f0, spc, ap
        wav = pyworld.synthesize(f0, spc, ap,
                                 fs, frame_period=shiftms)

        return wav

    def synthesis_spc(self, f0, spc, ap, fs=16000, shiftms=5):
        """
        synthesis generates waveform from F0, mcep, aperiodicity

        Parameters
        ----------
        f0: array, shape (T, `1`)
          array of F0 sequence
        mcep: array, shape (T, `self.conf.dim`)
          array of mel-cepstrum sequence
        aperiodicity: array, shape (T, `fftlen / 2 + 1`)
          array of aperiodicity
        dim: int, optional
          Dimension of the mel-cepstrum sequence
        alpha: int, optional
          Parameter of all-path fileter for frequency transformation
        fs: int, optional
          Sampling frequency
        shiftms: int, optional
          Shift size for STFT

        Return
        ------
        wav: vector
          Synethesized waveform

        """

        # generate waveform using world vocoder with f0, spc, ap
        wav = pyworld.synthesize(f0, spc, ap,
                                 fs, frame_period=shiftms)

        return wav
