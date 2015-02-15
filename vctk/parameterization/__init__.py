# coding: utf-8

import numpy as np
import numpy.fft
import sptk

from vctk import Parameterizer, SpectrumEnvelopeParameterizer

"""
Speech paramterization
"""


class TransparentParameterizer(Parameterizer):

    """
    Do nothing on raw parameters. Just pass the input to output.
    """

    def __init__(self):
        pass

    def forward(self, raw):
        return raw

    def backward(self, param):
        return paarm


class MelCepstrumParameterizer(SpectrumEnvelopeParameterizer):

    """
    Mel-cepstrum paraemterization
    """

    def __init__(self,
                 order=40,
                 alpha=0.41,
                 fftlen=1024
                 ):
        self.order = order
        self.alpha = alpha
        self.fftlen = fftlen

    def forward(self, spectrum_envelope):
        """
        Spectrum envelope -> Mel-cepstrum
        """
        return spgram2mcgram(spectrum_envelope, self.order, self.alpha)

    def backward(self, mc):
        """
        Mel-cepstrum -> Spectrum envelope
        """
        return mcgram2spgram(mc, self.alpha, self.fftlen)


def sp2mc(spec, order, alpha):
    '''
    sp2mc converts raw spectrum envelop to mel-cepstrum

    H(ω) -> cₐ(m)

    Parameters
    ----------
    spec: array, shape (`fftlen / 2 +1`)
      specturm envelope

    order: int
      order of mel-cepstrum

    alpha: float
      frequency warping paramter (all-pass constant)

    Return
    ------
    mel-cepstrum: array, shape (`order + 1`)

    '''
    # H(ω) -> log(H(ω)^2)
    logperiodogram = 2. * np.log(spec)

    # log(H(ω)^2) -> c(m)
    c = np.real(np.fft.irfft(logperiodogram))
    c[0] /= 2.0

    # c(m) -> cₐ(m)
    return sptk.freqt(np.ascontiguousarray(c), order, alpha)


def mc2sp(mc, alpha, fftlen):
    """
    mc2sp reconstruct raw spectrum envelope from mel-cepstrum

    cₐ(m) -> H(ω)

    Parameters
    ----------
    mc: array, shape (`order + 1`)
      mel-cepstrum

    alpha: float
      frequency warping paramter

    fftlen: int
      fft length

    Return
    ------
    spectrum envelope: array, shape (`fftlen`>>1 + 1)

    """

    # cₐ(m) -> c(m)
    c = sptk.freqt(np.ascontiguousarray(mc), fftlen >> 1, -alpha)
    c[0] *= 2.0

    symc = np.resize(c, fftlen)
    for i in range(1, len(c)):
        symc[-i] = c[i]
    assert symc[1] == symc[-1]

    # c(m) -> log(H(ω)^2) -> log(H(ω)) -> H(ω)
    return np.exp(np.real(np.fft.rfft(symc)) / 2)


def spgram2mcgram(spectrogram, order, alpha):
    """
    spgram2mcgram converts array of raw spectrum envelope to array of
    mel-cepstrum

    Parameters
    ----------
    spectrogram: array, shape (T, `fftlen / 2 +1`)
      array of specturm envelope

    order: int
      order of mel-cepstrum

    alpha: float
      frequency warping paramter (all-pass constant)

    Return
    ------
    array of mel-cepstrum: array, shape (T, `order + 1`)

    """
    T = spectrogram.shape[0]
    mcgram = np.zeros((T, order + 1))
    for t in range(T):
        mcgram[t] = sp2mc(spectrogram[t], order, alpha)
    return mcgram


def mcgram2spgram(mcgram, alpha, fftlen):
    """
    mc2sp converts array of  mel-cepstrum to array of spectrum envelope

    Parameters
    ----------
    mcgram: array, shape (`T`, `order + 1`)
      array of mel-cepstrum

    alpha: float
      frequency warping paramter

    fftlen: int
      fft length

    Return
    ------
    array of spectrum envelope: array, shape (`T`, `fftlen`>>1 + 1)

    """
    T = mcgram.shape[0]
    spectrogram = np.zeros((T, (fftlen >> 1) + 1))
    for t in range(T):
        spectrogram[t] = mc2sp(mcgram[t], alpha, fftlen)
    return spectrogram
