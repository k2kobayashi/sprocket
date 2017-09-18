# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
from sprocket.util import HDF5


def read_feats(listf, h5dir, ext='mcep'):
    """HDF5 handler
    Create list consisting of arrays listed in the list

    Parameters
    ---------
    listf : str,
        Path of list file

    h5dir : str,
        Path of hdf5 directory

    ext : str,
        `mcep` : mel-cepstrum
        `f0` : F0

    Returns
    ---------
    datalist : list of arrays

    """

    datalist = []
    with open(listf, 'r') as fp:
        for line in fp:
            f = line.rstrip()
            h5f = os.path.join(h5dir, f + '.h5')
            h5 = HDF5(h5f, mode='r')
            datalist.append(h5.read(ext))
            h5.close()

    return datalist
