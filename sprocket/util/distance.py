# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np


def melcd(array1, array2):
    """Normalized mel-cepstrum distortion over time

    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`)
        Arrays of original and target.

    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion

    """
    assert array1.shape == array2.shape

    if array1.ndim == 2:
        # array based melcd calculation
        diff = array1 - array2
        mcd = 10.0 / np.log(10) \
            * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    elif array1.ndim == 1:
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
    else:
        raise("Dimension mismatsh")

    return mcd
