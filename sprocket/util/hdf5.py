#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# hdf5.py
#   First ver.: 2017-06-07
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
handle hdf5 files

"""

import os
import h5py


class HDF5(object):

    """
    Handle HDF5 format file

    TODO:

    Attributes
    ----------
    """

    def __init__(self, fpath, mode=None):
        self.fpath = fpath
        self.dirname, _ = os.path.split(self.fpath)

        if mode == None:
            raise("Please specify the mode.")
        else:
            self.mode = mode

        if self.mode == "w":
            # create directory if not exist
            if not os.path.exists(self.dirname):
                os.makedirs(self.dirname)
            # file check
            if os.path.exists(self.fpath):
                print("overwrite because HDF5 file already exists.")
        elif self.mode == "r":
            if not os.path.exists(self.fpath):
                raise("h5 does not exist.")

        # open hdf5 file to fpath
        self.h5 = h5py.File(self.fpath, self.mode)

    def read(self, ext=None):
        if ext == None:
            raise("Please specify an extention.")

        if self.mode != "r":
            raise("mode should be 'r'")
        return self.h5[ext].value

    def save(self, data, ext=None):
        if ext == None:
            raise("Please specify an extention.")
        if self.mode != "w":
            raise("mode should be 'w'")

        self.h5.create_dataset(ext, data=data)
        self.h5.flush()

    def close(self):
        self.h5.close()


def main():
    pass


if __name__ == '__main__':
    main()
