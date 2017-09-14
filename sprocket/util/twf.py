# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import numpy as np
from dtw import dtw
from fastdtw import fastdtw

from sprocket.util import melcd


def estimate_twf(orgdata, tardata, distance='melcd', normalflag=False):
    """time warping function estimator

    Parameters
    ---------
    orgdata : array, shape(`T_org`, `dim`)
        Array of source feature
    tardata : array, shape(`T_tar`, `dim`)
        Array of target feature
    distance : str, optional
        distance function
        `melcd` : mel-cepstrum distortion
    fastflag : bool, optional
        Use fastdtw instead of dtw

    Returns
    ---------
    twf : array, shape(`2`, `T`)
        Time warping function between original and target
    """

    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y, vector=True)
    else:
        raise('other distance metrics does not support.')

    if normalflag:
        _, _, _, path = dtw(orgdata, tardata, distance_func)
    else:
        _, path = fastdtw(orgdata, tardata, dist=distance_func)
        twf = np.array(path).T

    return twf
