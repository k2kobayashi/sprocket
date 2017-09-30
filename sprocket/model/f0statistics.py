# -*- coding: utf-8 -*-

import numpy as np


class F0statistics(object):
    """F0 statistics class
    Estimate F0 statistics and convert F0

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

        f0stats = np.array([np.mean(f0s), np.std(f0s)])
        return f0stats

    def convert(self, f0, orgf0stats, tarf0stats):
        """Convert F0 based on F0 statistics

        Parameters
        ---------
        f0 : array, shape(`T`, `1`)
            Array of F0 sequence
        orgf0stats, shape (`[mean, std]`)
            Vector of mean and standard deviation of logarithmic F0 for original speaker
        tarf0stats, shape (`[mean, std]`)
            Vector of mean and standard deviation of logarithmic F0 for target speaker

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
        cvf0[nonzero_indices] = np.exp((tarf0stats[1] / orgf0stats[1]) *
                                       (np.log(f0[nonzero_indices]) -
                                        orgf0stats[0]) + tarf0stats[0])

        return cvf0
