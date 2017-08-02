# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import

import os
import numpy as np


class GV (object):

    """A global variance (GV) statistics class
    This class offers the estimation of GV statistics and
    postfiltering based on the GV statistics

    Attributes
    ---------
    gvstats : shape (`2`, `dim`)
        Array of mean and standard deviation of GV

    """

    def __init__(self):
        pass

    def estimate(self, datalist):
        """Estimate GV statistics from list of data

        Parameters
        ---------
        datalist : list, shape ('num_data')
            List of several data ([T, dim]) sequence

        Returns
        ---------
        gvstats : array, shape (`2`, `dim`)
            Array of mean and standard deviation fo GV
        """

        n_files = len(datalist)

        var = []
        for i in range(n_files):
            data = datalist[i]
            var.append(np.var(data, axis=0))

        # calculate vm and vv
        vm = np.mean(np.array(var), axis=0)
        vv = np.var(np.array(var), axis=0)
        gvstats = np.r_[vm, vv]
        self.gvstats = gvstats.reshape(2, len(vm))

        return

    def save(self, fpath):
        """Save GV statistics into fpath as binary

        Parameters
        ---------
        fpath: str,
            File path of GV statistics

        gvstats: array, shape (`2`, `dim`)
            GV statistics

        """

        if self.gvstats is None:
            raise('gvstats does not calculated')

        if not os.path.exists(os.path.dirname(fpath)):
            os.makedirs(os.path.dirname(fpath))

        self.gvstats.tofile(fpath)

    def open_from_file(self, fpath):
        """Open GV statistics from binary file

        Parameters
        ---------
        fpath: str,
            File path of GV statistics

        """

        # read gv from binary
        gv = np.fromfile(fpath)
        dim = len(gv) // 2
        self.gvstats = gv.reshape(2, dim)

        return

    def postfilter(self, data, startdim=1):
        """Perform postfilter based on GV statistics into data

        Parameters
        ---------
        data : array, shape (`T`, `dim`)
            Array of data sequence

        startdim : int, optional
            Start dimension to perform GV postfilter

        Returns
        ---------
        filtered_data : array, shape (`T`, `data`)
            Array of GV postfiltered data sequence

        """

        # get length and dimension
        T, dim = data.shape
        assert self.gvstats is not None
        assert dim == self.gvstats.shape[1]

        # calculate statics of input data
        datamean = np.mean(data, axis=0)
        datavar = np.var(data, axis=0)

        # perform GV postfilter
        filtered = np.sqrt(self.gvstats[0, startdim:] / datavar[startdim:]) * \
            (data[:, startdim:] - datamean[startdim:]) + datamean[startdim:]

        filtered_data = np.c_[data[:, :startdim], filtered]

        return filtered_data
