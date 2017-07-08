# -*- coding: utf-8 -*-

import numpy as np


def extfrm(npow, data, threshold=-20):
    """Extract frame under the threshold

    Parameters
    ----------
    npow : array, shape (`T`)
        Vector of normalized power sequence.

    data: array, shape (`T`, `dim`)
        Array of input data

    threshold: scala, optional
        Scala of power threshold

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
