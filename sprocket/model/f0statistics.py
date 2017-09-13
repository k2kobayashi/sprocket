# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import numpy as np


class F0statistics(object):

    """F0 statistics class
    Estimate F0 statistics and convert F0

    Attributes
    ---------
    orgf0stats, shape (`[mean, std]`)
        Vector of mean and standard deviation of logarithmic F0 for original speaker
    tarf0stats, shape (`[mean, std]`)
        Vector of mean and standard deviation of logarithmic F0 for target speaker

    """

    def __init__(self):
        pass

    def estimate(self, f0list):
        """Estimate F0 statistics from list of f0

        Parameters
        ---------
        f0list : list, shape('f0num')
            List of several F0 sequence

        Returns
        ---------
        f0stats : array, shape(`[mean, std]`)
            Values of mean and standard deviation for logarithmic F0

        """

        n_files = len(f0list)
        for i in range(n_files):
            f0 = f0list[i]
            nonzero_indices = np.nonzero(f0)
            if i == 0:
                f0s = np.log(f0[nonzero_indices])
            else:
                f0s = np.r_[f0s, np.log(f0[nonzero_indices])]

        self.f0stats = np.array([np.mean(f0s), np.std(f0s)])

        return

    def save(self, fpath):
        """Save f0 statistics into fpath as binary

        Parameters
        ---------
        fpath : str,
            File path of F0 statistics

        """

        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))
        self.f0stats.tofile(fpath)

    def open_from_file(self, orgfile, tarfile):
        """Open F0 statistics from file

        Parameters
        ---------
        orgfile : str
            File path of F0 statistics for original speaker
        tarfile : str
            File path of F0 statistics for target speaker

        """

        # read f0 statistics of source and target from binary
        self.orgf0stats = np.fromfile(orgfile)
        self.tarf0stats = np.fromfile(tarfile)

    def convert(self, f0):
        """Convert F0 based on F0 statistics

        Parameters
        ---------
        f0 : array, shape(`T`, `1`)
            Array of F0 sequence

        Returns
        ---------
        cvf0 : array, shape(`T`, `1`)
            Array of converted F0 sequence

        """

        # get length and dimension
        T = len(f0)

        # perform f0 conversion
        cvf0 = np.zeros(T)

        nonzero_indices = f0 > 0
        cvf0[nonzero_indices] = np.exp((self.tarf0stats[1] / self.orgf0stats[1]) *
                                       (np.log(f0[nonzero_indices]) -
                                        self.orgf0stats[0]) + self.tarf0stats[0])

        return cvf0
