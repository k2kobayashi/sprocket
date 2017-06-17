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

from sprocket.util.yml import PairYML
from sprocket.util.hdf5 import open_h5files, close_h5files


class F0statistics (object):

    def __init__(self, yml):
        # read pair-dependent yml file
        self.conf = PairYML(yml)

        # open h5list files
        self.h5s = open_h5files(yml, mode='tr')
        self.num_files = len(self.h5s)

    def estimate(self):
        otflags = [0, 1]
        for otflag in otflags:
            if otflag == 1:
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


def main():
    pass


if __name__ == '__main__':
    main()
