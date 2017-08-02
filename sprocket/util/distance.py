# -*- coding: utf-8 -*-



import numpy as np


def melcd(vec1, vec2):
    """Calculate mel-cepstrum distortion

    Parameters
    ----------
    vec1 : array, shape (`dim`)
        Vector of original

    vec2 : array, shape (`dim`)
        Vector of Target

    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion

    """
    assert len(vec1) == len(vec2)

    diff = vec1 - vec2
    mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))

    return mcd


def normalized_melcd(array1, array2):
    """Normalized mel-cepstrum distortion over time

    Parameters
    ----------
    array1 : array, shape (`T`, `dim`)
        Array of Original.

    array2 : array, shape (`T`, `dim`)
        Array of Target

    Returns
    -------
    norm_mcd : scala, number > 0
        Scala of normalized mel-cepstrum distortion.

    """
    assert array1.shape == array2.shape

    diff = array1 - array2

    norm_mcd = 10.0 / np.log(10) \
        * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))

    return norm_mcd
