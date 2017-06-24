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
Calculate log F0 mean and variance of several feature sequence

"""

import os
import numpy as np

from sprocket.util.hdf5 import open_h5files, close_h5files


class F0statistics (object):

    def __init__(self, conf):
        # read pair-dependent yml file
        self.conf = conf

        # open h5list files
        self.h5s = open_h5files(conf, mode='tr')
        self.num_files = len(self.h5s)

    def estimate(self):
        otflags = [0, 1]
        for otflag in otflags:
            if otflag == 0:
                spkr = 'org'
            else:
                spkr = 'tar'

            for i in range(self.num_files):
                nonzeroidx = np.nonzero(self.h5s[i][otflag].read('f0'))
                f0 = self.h5s[i][otflag].read('f0')
                if i == 0:
                    f0s = np.log(f0[nonzeroidx])
                else:
                    f0s = np.r_[f0s, np.log(f0[nonzeroidx])]

            f0stats = np.array([np.mean(f0s), np.std(f0s)])

            f0statspath = self.conf.pairdir + '/stats/' + spkr + '.f0stats'
            if not os.path.exists(os.path.dirname(f0statspath)):
                os.makedirs(os.path.dirname(f0statspath))
            fp = open(f0statspath, 'w')
            fp.write(f0stats)
            fp.close()

            print('F0 statistics estimation for ' + spkr + ' has been done.')

        # close h5class
        close_h5files(self.h5s)

        return

    def read_statistics(self, orgstatsf, tarstatsf):
        # read f0 statistics of source and target from binary
        orgf0stats = np.fromfile(orgstatsf, dtype='d')
        tarf0stats = np.fromfile(tarstatsf, dtype='d')

        self.omean = orgf0stats[0]
        self.ostd = orgf0stats[1]
        self.tmean = tarf0stats[0]
        self.tstd = tarf0stats[1]

        return

    def convert_f0(self, f0):
        assert self.omean, self.ostd is not None
        assert self.tmean, self.tstdis is not None

        # get length and dimension
        T = len(f0)

        # perform f0 conversion
        cvf0 = np.zeros(T)

        nonzero_indices = f0 > 0
        cvf0[nonzero_indices] = (f0[nonzero_indices] - self.omean) * \
            self.tstd / self.ostd + self.tmean

        return cvf0


def main():
    pass


if __name__ == '__main__':
    main()
