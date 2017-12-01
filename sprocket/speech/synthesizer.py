# -*- coding: utf-8 -*-

import numpy as np
import pyworld
import pysptk
from pysptk.synthesis import MLSADF


class Synthesizer(object):
    """
    Speech synthesizer with several acoustic features

    Parameters
    ----------
    fs: int, optional
        Sampling frequency
        Default set to 16000
    fftl: int, optional
        Frame Length of STFT
        Default set to 1024
    shiftms: int, optional
        Shift size for STFT
        Default set to 5
    """

    def __init__(self, fs=16000, fftl=1024, shiftms=5):
        self.fs = fs
        self.fftl = fftl
        self.shiftms = shiftms

        return

    def synthesis(self, f0, mcep, ap, rmcep=None, alpha=0.42):
        """synthesis generates waveform from F0, mcep, aperiodicity

        Parameters
        ----------
        f0 : array, shape (`T`, `1`)
            array of F0 sequence
        mcep : array, shape (`T`, `dim`)
            array of mel-cepstrum sequence
        ap : array, shape (`T`, `fftlen / 2 + 1`) or (`T`, `dim_codeap`)
            array of aperiodicity or code aperiodicity
        rmcep : array, optional, shape (`T`, `dim`)
            array of reference mel-cepstrum sequence
            Default set to None
        alpha : int, optional
            Parameter of all-path transfer function
            Default set to 0.42

        Returns
        ----------
        wav: array,
            Synethesized waveform

        """

        if rmcep is not None:
            # power modification
            mcep = mod_power(mcep, rmcep, alpha=alpha)

        if ap.shape[1] < self.fftl // 2 + 1:
            # decode codeap to ap
            ap = pyworld.decode_aperiodicity(ap, self.fs, self.fftl)

        # mcep into spc
        spc = pysptk.mc2sp(mcep, alpha, self.fftl)

        # generate waveform using world vocoder with f0, spc, ap
        wav = pyworld.synthesize(f0, spc, ap,
                                 self.fs, frame_period=self.shiftms)

        return wav

    def synthesis_diff(self, x, diffmcep, rmcep=None, alpha=0.42):
        """filtering with a differential mel-cesptrum

        Parameters
        ----------
        x : array, shape (`samples`)
            array of waveform sequence
        diffmcep : array, shape (`T`, `dim`)
            array of differential mel-cepstrum sequence
        rmcep : array, shape (`T`, `dim`)
            array of reference mel-cepstrum sequence
            Default set to None
        alpha : float, optional
            Parameter of all-path transfer function
            Default set to 0.42

        Return
        ----------
        wav: array, shape (`samples`)
            Synethesized waveform

        """

        x = x.astype(np.float64)
        dim = diffmcep.shape[1] - 1
        shiftl = int(self.fs / 1000 * self.shiftms)

        if rmcep is not None:
            # power modification
            diffmcep = mod_power(rmcep + diffmcep, rmcep, alpha=alpha) - rmcep

        b = np.apply_along_axis(pysptk.mc2b, 1, diffmcep, alpha)
        assert np.isfinite(b).all()

        mlsa_fil = pysptk.synthesis.Synthesizer(
            MLSADF(dim, alpha=alpha), shiftl)
        wav = mlsa_fil.synthesis(x, b)

        return wav

    def synthesis_spc(self, f0, spc, ap):
        """synthesis generates waveform from F0, mcep, ap

        Parameters
        ----------
        f0 : array, shape (`T`, `1`)
          array of F0 sequence
        spc : array, shape (`T`, `fftl // 2 + 1`)
          array of mel-cepstrum sequence
        ap : array, shape (`T`, `fftl // 2 + 1`)
          array of aperiodicity

        Return
        ------
        wav: vector, shape (`samples`)
          Synethesized waveform

        """

        # generate waveform using world vocoder with f0, spc, ap
        wav = pyworld.synthesize(f0, spc, ap,
                                 self.fs, frame_period=self.shiftms)

        return wav


def mod_power(cvmcep, rmcep, alpha=0.42, irlen=1024):
    """Power modification based on inpulse responce

    Parameters
    ----------
    cvmcep : array, shape (`T`, `dim`)
        array of converted mel-cepstrum
    rmcep : array, shape (`T`, `dim`)
        array of reference mel-cepstrum
    alpha : float, optional
        All-path filter transfer function
        Default set to 0.42
    irlen : int, optional
        Length for IIR filter
        Default set to 1024

    Return
    ------
    modified_cvmcep : array, shape (`T`, `dim`)
        array of power modified converted mel-cepstrum

    """

    if rmcep.shape != cvmcep.shape:
        raise ValueError("The shapes of the converted and \
                         reference mel-cepstrum are different: \
                         {} / {}".format(cvmcep.shape, rmcep.shape))

    cv_e = pysptk.mc2e(cvmcep, alpha=alpha, irlen=irlen)
    r_e = pysptk.mc2e(rmcep, alpha=alpha, irlen=irlen)

    dpow = np.log(r_e / cv_e) / 2

    modified_cvmcep = np.copy(cvmcep)
    modified_cvmcep[:, 0] += dpow

    return modified_cvmcep
