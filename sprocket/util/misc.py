# -*- coding: utf-8 -*-

import os
import numpy as np

from . import HDF5, extfrm, static_delta


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


def extsddata(data, npow, power_threshold=-20):
    """Get power extract static and delta feature vector

    Paramters
    ---------
    data : array, shape (`T`, `dim`)
        Acoustic feature vector
    npow : array, shape (`T`)
        Normalized power vector
    power_threshold : float, optional,
        Power threshold
        Default set to -20

    Returns
    -------
    extsddata : array, shape (`T_new` `dim * 2`)
        Silence remove static and delta feature vector

    """

    extsddata = extfrm(static_delta(data), npow,
                       power_threshold=power_threshold)
    return extsddata


def transform_jnt(array_list):
    num_files = len(array_list)
    for i in range(num_files):
        if i == 0:
            jnt = array_list[i]
        else:
            jnt = np.r_[jnt, array_list[i]]
    return jnt


def list_lengths_are_all_same(first_path, *remain_paths):
    """Check whether the lengths of list files are all same.

    Parameters
    ----------
    first_path : str or path-like object
    remain_paths : list of str or path-like object

    Returns
    -------
    list_lengths_are_all_same : bool
        `True` if all of the numbers of the lengths of the given files are same.

    Notes
    ----
        The length of each file is the same as the Unix command `wc -w`.
    """

    def count_words_in_file(path):
        """Counts the number of words in a file.

        Parameters
        ----------
        path : str or path-like object
            The path of the file the number of words of which you want to know.

        Returns
        -------
        n_words : int
            The number of words in the file whose path is `path`.
        """
        with open(str(path)) as handler:
            words = len(
                handler.read().split())  # when space in path, bug appears
        return words

    n_words_in_first_file = count_words_in_file(first_path)
    return all((count_words_in_file(path) == n_words_in_first_file
                for path in remain_paths))
