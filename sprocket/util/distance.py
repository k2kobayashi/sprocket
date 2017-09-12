# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np


def melcd(array1, array2, vector=False):
    """Normalized mel-cepstrum distortion over time

    Parameters
    ----------
    array1, array2 : array, shape (`T`, `dim`)
        Arrays of original and target.
    vec : bool, optional
        Calculate melcd only a single vector
        `False` : calculate melcd as array
        `True` : calculate melcd as vector
            vector should be (`dim`)

    Returns
    -------
    mcd : scala, number > 0
        Scala of normalized mel-cepstrum distortion

    """

    if not vector:
        # array based melcd calculation
        assert array1.shape == array2.shape

        diff = array1 - array2
        norm_mcd = 10.0 / np.log(10) \
            * np.mean(np.sqrt(2.0 * np.sum(diff ** 2, axis=1)))
        return norm_mcd
    else:
        # vector based melcd calculation
        assert len(array1) == len(array2)

        diff = array1 - array2
        mcd = 10.0 / np.log(10) * np.sqrt(2.0 * np.sum(diff ** 2))
        return mcd
