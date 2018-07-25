# -*- coding: utf-8 -*-

import numpy as np
from dtw import dtw
from fastdtw import fastdtw
from dtw_c import dtw_c

from sprocket.util import melcd


def estimate_twf(orgdata, tardata, distance='melcd', fast=True, otflag=None):
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
    otflag : str,
        Perform alignment into either original or target length
        `org` : align into original length
        `tar` : align into target length
        Default set to None

    Returns
    ---------
    twf : array, shape(`2`, `T`)
        Time warping function between original and target
    """

    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y)
    else:
        raise ValueError('other distance metrics than melcd does not support.')

    if otflag is None:
        # use dtw or fastdtw
        if fast:
            _, path = fastdtw(orgdata, tardata, dist=distance_func)
            twf = np.array(path).T
        else:
            _, _, _, twf = dtw(orgdata, tardata, distance_func)
    else:
        # use dtw_c to align target/original feature vector
        ldim = orgdata.shape[1] - 1
        if otflag == 'org':
            _, twf, _, _ = dtw_c.dtw_org_to_trg(tardata, orgdata,
                                                0, ldim, 5.0, 100.0, 100.0)
        else:
            _, twf, _, _ = dtw_c.dtw_org_to_trg(orgdata, tardata,
                                                0, ldim, 5.0, 100.0, 100.0)
        twf[:, 1] = np.array(range(twf.shape[0]))  # replace target index by frame number
        twf = twf.T
        if otflag == 'org':
            twf = twf[::-1, :]  # swap cols
            assert twf.shape[0] == orgdata.shape[0]
        else:
            assert twf.shape[1] == tardata.shape[0]

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


def modify_twf(twf, otflag=None):
    """Align specified length

    Paramters
    ---------
    twf : array, shape(`2`, `T`)
        Time warping function between original and target
    otflag : str,
        Perform alignment into either original or target length
        `org` : align into original length
        `tar` : align into target length

    Returns
    -------
    mod_twf : array, shape (`2`, `T_new`)
        Time warping function of modified alignment

    """

    if otflag == 'org':
        of, indice = np.unique(twf[0], return_index=True)
        mod_twf = np.c_[of, twf[1][indice]].T
    elif otflag == 'tar':
        tf, indice = np.unique(twf[1], return_index=True)
        mod_twf = np.c_[twf[0][indice], tf].T
    return mod_twf
