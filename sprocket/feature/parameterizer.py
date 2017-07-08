# coding: utf-8

import numpy as np
import pysptk


def spgram2mcgram(spectrogram, order, alpha):
    """Convert spectrogram into mel-cepstrum sequence

    Converts array of raw spectrum envelope to array of mel-cepstrum

    Parameters
    ----------
    spectrogram : array, shape (`T`, `fftlen / 2 + 1`)
        Array of specturm envelope

    order: int
        Order of mel-cepstrum

    alpha: float
        Frequency warping paramter (all-pass constant)

    Return
    ------
    mcgram : array, shape (`T`, `order + 1`)

    """

    T = spectrogram.shape[0]
    mcgram = np.zeros((T, order + 1))
    for t in range(T):
        mcgram[t] = pysptk.sp2mc(spectrogram[t], order, alpha)

    return mcgram


def mcgram2spgram(mcgram, alpha, fftlen):
    """Convert mel-cepstrum sequence into spectrogram

    Converts array of mel-cepstrum to array of spectrum envelope

    Parameters
    ----------
    mcgram : array, shape (`T`, `order + 1`)
        Array of mel-cepstrum

    alpha : float
        Frequency warping paramter

    fftlen : int
        FFT length

    Return
    ------
    spectrogram : array, shape (`T`, `fftlen`>>1 + 1)

    """

    T = mcgram.shape[0]
    spectrogram = np.zeros((T, (fftlen >> 1) + 1))
    for t in range(T):
        spectrogram[t] = pysptk.mc2sp(mcgram[t], alpha, fftlen)

    return spectrogram


def spgram2npow(spectrogram):
    """Calculate normalized_melcd power sequence from spectrogram

    Parameters
    ----------
    spectrogram : array, shape (T, `fftlen / 2 + 1`)
        Array of specturm envelope

    Return
    ------
    npow : array, shape (`T`, `1`)
        Normalized power sequence

    """

    T = spectrogram.shape[0]
    npow = np.zeros(T)

    for t in range(T):
        npow[t] = spvec2pow(spectrogram[t])
    sumpow = sum(npow) / T

    for t in range(T):
        npow[t] = 10.0 * np.log10(npow[t] / sumpow)

    return npow


def spvec2pow(specvec):
    """Convert vector of spectrum envelope into normalized power

    Parameters
    ----------
    specvec : vector, shape (`fftlen / 2 + 1`)
        Cector of specturm envelope

    Return
    ------
    power : scala,
        Normalized power

    """

    # set FFT length
    fftl2 = len(specvec) - 1
    fftl = fftl2 * 2

    # specvec is not amplitude spectral |H(w)| but power spectral |H(w)|^2
    power = specvec[0] + specvec[fftl2]
    for k in range(1, fftl2):
        power += 2.0 * specvec[k]
    power /= fftl

    return power
