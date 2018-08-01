# -*- coding: utf-8 -*-

import os
import h5py


class HDF5(object):
    """HDF5 handler
    Offer the hdf5 format file handler

    Parameters
    ---------
    fpath : str,
        Path of hdf5 file

    mode : str,
        Open h5 as write and/or read mode
        `a` : open as read/write (Default)
        `w` : open as write only
        `r` : open as read only

    Attributes
    ---------
    h5 : hdf5 class

    """

    def __init__(self, fpath, mode='a'):
        self.fpath = fpath
        self.dirname, self.filename = os.path.split(self.fpath)
        self.flbl, _ = os.path.splitext(self.filename)

        if mode is None:
            raise ValueError("Please specify the mode.")
        else:
            self.mode = mode

        # create directory if not exist
        if self.mode == 'w' or self.mode == 'a':
            # create directory if not exist
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)

        # warn overwriting
        if self.mode == 'w':
            if os.path.exists(self.fpath):
                print("overwrite: " + self.fpath)

        # check file existing for reading
        if self.mode == 'r':
            if not os.path.exists(self.fpath):
                raise FileNotFoundError(
                    "h5 file does not exist in " + self.fpath)

        # open hdf5 file to fpath
        self.h5 = h5py.File(self.fpath, self.mode)

    def read(self, ext=None):
        """Read vector or array from h5 file

        Parameters
        ---------
        ext : str
            File extention including h5 file

        Returns
        -------
        array : array,
            Array of hdf5 packed data
        """

        if ext is None:
            raise ValueError("Please specify an existing extention.")

        return self.h5[ext].value

    def save(self, data, ext=None):
        """Write vector or array into h5 file

        Parameters
        ---------
        data :
            Vector or array will be wrote into h5 file

        ext: str
            File label of saved file

        """

        # remove if 'ext' already exist
        if ext in self.h5.keys():
            del self.h5[ext]

        self.h5.create_dataset(ext, data=data)
        self.h5.flush()

        return

    def close(self):
        self.h5.close()

        return
