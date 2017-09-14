# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np


def extfrm(data, npow, threshold=-20):
    """Extract frame over the power threshold

    Parameters
    ----------
    data: array, shape (`T`, `dim`)
        Array of input data
    npow : array, shape (`T`)
        Vector of normalized power sequence.
    threshold: scala, optional
        Scala of power threshold [dB]

    Returns
    -------
    data: array, shape (`T_ext`, `dim`)
        Remaining data after extracting frame
        `T_ext` <= `T`

    """

    T = data.shape[0]
    if T != len(npow):
        raise("Length of two vectors is different.")

    valid_index = np.where(npow > threshold)

    return data[valid_index]
