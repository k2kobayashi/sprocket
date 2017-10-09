# -*- coding: utf-8 -*-

import numpy as np


class GV (object):
    """A global variance (GV) statistics class
    Estimate statistics and perform postfilter based on
    the GV statistics

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
            Array of mean and standard deviation for GV
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
        gvstats = gvstats.reshape(2, len(vm))

        return gvstats

    def postfilter(self, data, gvstats, cvgvstats=None, alpha=1.0, startdim=1):
        """Perform postfilter based on GV statistics into data

        Parameters
        ---------
        data : array, shape (`T`, `dim`)
            Array of data sequence
        gvstats: array, shape (`2`, `dim`)
            Array of mean and variance for target GV
        cvgvstats: array, shape (`2`, `dim`), optional
            Array of mean and variance for converted GV
            This option replaces the mean variance of the given data
            into the mean variance estimated from training data.
        alpha : float, optional
            Morphing coefficient between GV transformed data and data.
            .. math::
               alpha * gvpf(data) + (1 - alpha) * data
            Default set to 1.0
        startdim : int, optional
            Start dimension to perform GV postfilter

        Returns
        ---------
        filtered_data : array, shape (`T`, `data`)
            Array of GV postfiltered data sequence

        """

        # get length and dimension
        T, dim = data.shape
        assert gvstats is not None
        assert dim == gvstats.shape[1]

        # calculate statics of input data
        datamean = np.mean(data, axis=0)

        if cvgvstats is None:
            # use variance of the given data
            datavar = np.var(data, axis=0)
        else:
            # use variance of trained gv stats
            datavar = cvgvstats[0]

        # perform GV postfilter
        filtered = np.sqrt(gvstats[0, startdim:] / datavar[startdim:]) * \
            (data[:, startdim:] - datamean[startdim:]) + datamean[startdim:]

        filtered_data = np.c_[data[:, :startdim], filtered]

        return alpha * filtered_data + (1 - alpha) * data
