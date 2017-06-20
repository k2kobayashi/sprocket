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

        # open h5list files
        self.h5s = open_h5files(conf, mode='tr')
        self.num_files = len(self.h5s)

    def estimate(self, feature='mcep'):
        otflags = [0, 1]
        for otflag in otflags:
            if otflag == 1:
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
            gv = np.c_[vm, vv].T

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

    def gv_postfilter(self):
        pass


def main():
    pass


if __name__ == '__main__':
    main()
