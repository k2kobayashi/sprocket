#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# npow.py
#   First ver.: 2017-06-07
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
calculate normalized power from spectrogram


"""

import numpy as np

def spgram2npow(spectrogram):
    """
    spgram2npow converts array of spectrum envelopw into that of normalized power

    Parameters
    ----------
    spectrogram: array, shape (T, `fftlen / 2 + 1`)
      array of specturm envelope

    Return
    ------
    array of normalized power: array, shape (`T`, `1`)

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
    """
    spvec2pow converts vector of spectrum envelopw into a scala of power

    Parameters
    ----------
    specvec: vector, shape (`fftlen / 2 + 1`)
      vector of specturm envelope

    Return
    ------
    scala of power in a frame

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

def main():
    pass


if __name__ == '__main__':
    main()
