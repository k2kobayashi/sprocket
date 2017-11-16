# -*- coding: utf-8 -*-

import numpy as np
from dtw import dtw
from fastdtw import fastdtw

from sprocket.util import melcd


def estimate_twf(orgdata, tardata, distance='melcd', fast=True):
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
    fast : bool, optional
        Use fastdtw instead of dtw
        Default set to `True`

    Returns
    ---------
    twf : array, shape(`2`, `T`)
        Time warping function between original and target
    """

    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y)
    else:
        raise ValueError('other distance metrics than melcd does not support.')

    if fast:
        _, path = fastdtw(orgdata, tardata, dist=distance_func)
        twf = np.array(path).T
    else:
        _, _, _, twf = dtw(orgdata, tardata, distance_func)

    return twf


def align_data(org_data, tar_data, twf):
    """Get aligned joint feature vector

    Paramters
    ---------
    orgdata : array, shape (`T_org`, `dim_org`)
        Acoustic feature vector of original speaker
    tardata : array, shape (`T_tar`, `dim_tar`)
        Acoustic feature vector of target speaker
    twf : array, shape (`2`, `T`)
        Time warping function between original and target

    Returns
    -------
    jdata : array, shape (`T_new` `dim_org + dim_tar`)
        Joint feature vector between source and target

    """

    jdata = np.c_[org_data[twf[0]], tar_data[twf[1]]]
    return jdata
