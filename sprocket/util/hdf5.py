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

from sprocket.util.yml import PairYML


def open_h5files(conf, mode='tr'):
    # read h5 files
    h5list = []
    if mode == 'tr':
        num_files = len(conf.trfiles)
        for i in range(num_files):
            # open acoustic features
            orgh5 = HDF5(
                conf.h5dir + '/' + conf.trfiles[i][0] + '.h5', mode="r")
            tarh5 = HDF5(
                conf.h5dir + '/' + conf.trfiles[i][1] + '.h5', mode="r")
            h5list.append([orgh5, tarh5])
    elif mode == 'ev':
        num_files = len(conf.evfiles)
        for i in range(num_files):
            # open acoustic features
            orgh5 = HDF5(
                conf.h5dir + '/' + conf.evfiles[i] + '.h5', mode="r")
            h5list.append(orgh5)
    else:
        raise('other mode does not support')

    return h5list


def close_h5files(h5list, mode='tr'):
    # close hdf5 files
    for i in range(len(h5list)):
        if mode == 'tr':
            h5list[i][0].close()
            h5list[i][1].close()
        else:
            h5list[i].close()
    return


class HDF5(object):

    """
    Handle HDF5 format file for a file

    TODO:

    Attributes
    ----------
    """

    def __init__(self, fpath, mode=None):
        self.fpath = fpath
        self.dirname, self.filename = os.path.split(self.fpath)
        self.flbl, _ = os.path.splitext(self.filename)

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
                print("overwrite: " + self.fpath)
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
