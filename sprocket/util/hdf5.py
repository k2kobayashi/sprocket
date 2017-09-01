# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import h5py


class HDF5(object):

    """HDF5 handler
    This class offers the hdf5 format file for acoustic features

    Parameters
    ---------
    fpath : str,
        Path of hdf5 file

    mode : str,
        Open h5 as write or read mode
        `w` : open as write
        `r` : open as read

    Attributes
    ---------
    h5 : hdf5 class

    """

    def __init__(self, fpath, mode=None):
        self.fpath = fpath
        self.dirname, self.filename = os.path.split(self.fpath)
        self.flbl, _ = os.path.splitext(self.filename)

        if mode == None:
            raise("Please specify the mode.")
        else:
            self.mode = mode

        if self.mode == 'w':
            # create directory if not exist
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
            # file check
            if os.path.exists(self.fpath):
                print("overwrite: " + self.fpath)
        elif self.mode == 'r':
            if not os.path.exists(self.fpath):
                raise "h5 file does not exist in " + self.fpath

        # open hdf5 file to fpath
        self.h5 = h5py.File(self.fpath, self.mode)

    def read(self, ext=None):
        """Read vector or array from h5 file

        Parameters
        ---------
        ext : str
            File extention including h5 file

        """

        if ext == None:
            raise("Please specify an extention.")

        if self.mode != 'r':
            raise("mode should be 'r'")

        return self.h5[ext].value

    def save(self, data, ext=None):
        """Write vector or array into h5 file

        Parameters
        ---------
        data :
            Vector or array will be wrote into h5 file

        ext: str
            File extention or file label

        """

        if ext is None:
            raise("Please specify an extention.")
        if self.mode != 'w':
            raise("mode should be 'w'")

        self.h5.create_dataset(ext, data=data)
        self.h5.flush()

        return

    def close(self):
        self.h5.close()

        return


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
        files = fp.readlines()

    for f in files:
        f = f.rstrip()
        h5f = h5dir + '/' + f + '.h5'
        h5 = HDF5(h5f, mode='r')
        datalist.append(h5.read(ext))
        h5.close()

    return datalist
