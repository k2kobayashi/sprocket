#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# gv.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
Calculate GV of several feature sequence

"""

import os
import numpy as np

from sprocket.util.hdf5 import open_h5files, close_h5files


class GV (object):

    def __init__(self, conf):
        # read pair-dependent yml file
        self.conf = conf

    def estimate(self, feature='mcep'):
        # open h5list files
        self.h5s = open_h5files(self.conf, mode='tr')
        self.num_files = len(self.h5s)

        otflags = [0, 1]
        for otflag in otflags:
            if otflag == 0:
                spkr = 'org'
            else:
                spkr = 'tar'

            var = []
            for i in range(self.num_files):
                mcep = self.h5s[i][otflag].read(feature)
                var.append(np.var(mcep, axis=0))

            # calculate vm and vv
            vm = np.mean(np.array(var), axis=0)
            vv = np.var(np.array(var), axis=0)
            gv = np.r_[vm, vv]
            gv = gv.reshape(2, len(vm))

            gvpath = self.conf.pairdir + '/stats/' + spkr + '.gv'
            if not os.path.exists(os.path.dirname(gvpath)):
                os.makedirs(os.path.dirname(gvpath))
            fp = open(gvpath, 'w')
            fp.write(gv)
            fp.close()

            print('GV estimation of ' + feature +
                  ' for ' + spkr + ' has been done.')

        # close h5class
        close_h5files(self.h5s)

        return

    def read_statistics(self, fpath):
        # read gv from binary
        gv = np.fromfile(fpath, dtype='d')
        dim = len(gv) / 2
        self.gv = gv.reshape(2, dim)
        return

    def apply_gvpostfilter(self, data, startdim=1):
        # get length and dimension
        T, dim = data.shape
        assert self.gv is not None
        assert dim + startdim == self.gv.shape[1]

        # calculate statics of input data
        datamean = np.mean(data, axis=0)
        datavar = np.var(data, axis=0)

        # perform GV postfilter
        filtered_data = np.sqrt(self.gv[0, startdim:] / datavar) * \
            (data - datamean) + datamean

        return filtered_data


def main():
    pass


if __name__ == '__main__':
    main()
