# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
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
        """synthesis generates waveform from F0, mcep, ap

        Parameters
        ----------
        f0: array, shape (T, `1`)
          array of F0 sequence
        mcep: array, shape (T, `self.conf.dim`)
          array of mel-cepstrum sequence
        ap : array, shape (T, `fftlen / 2 + 1`)
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


def mod_power(cvmcep, rmcep, alpha=0.42):
    """synthesis generates waveform from F0, mcep, ap

    Parameters
    ----------
    cvmcep : array, shape (`T`, `dim`)
        array of converted mel-cepstrum
    rmcep : array, shape (`T`, `dim`)
        array of reference mel-cepstrum
    alpha : float, optional
        All-path filter transfer function

    Return
    ------
    modified_cvmcep : array, shape (`T`, `dim`)
        array of power modified converted mel-cepstrum

    """

    assert rmcep.shape == cvmcep.shape

    r_spc = pysptk.mc2sp(rmcep, alpha, 513)
    cv_spc = pysptk.mc2sp(cvmcep, alpha, 513)

    r_pow = np.mean(np.log(np.sqrt(r_spc)), axis=1)
    cv_pow = np.mean(np.log(np.sqrt(cv_spc)), axis=1)
    dpow = r_pow - cv_pow

    modified_cvmcep = cvmcep
    modified_cvmcep[:, 0] += dpow

    return modified_cvmcep
