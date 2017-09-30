# -*- coding: utf-8 -*-

import numpy as np


def melcd(array1, array2):
    """Calculate mel-cepstrum distortion

    Calculate mel-cepstrum distortion between the arrays.
    This function assumes the shapes of arrays are same.

    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`) or shape (`dim`)
        Arrays of original and target.

    Returns
    -------
    mcd : scala, number > 0
        Scala of mel-cepstrum distortion

    """
    if array1.shape != array2.shape:
        raise ValueError(
            "The shapes of both arrays are different \
            : {} / {}".format(array1.shape, array2.shape))

    if array1.ndim == 2:
        # array based melcd calculation
        diff = array1 - array2
        mcd = 10.0 / np.log(10) \
            * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
    elif array1.ndim == 1:
        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
    else:
        raise ValueError("Dimension mismatch")

    return mcd
