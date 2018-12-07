# -*- coding: utf-8 -*-

import numpy as np


class MS (object):
    """A modulation spectrum (MS) statistics class
    Estimate statistics and perform postfilter based on
    the MS statistics

    """

    def __init__(self):
        pass

    def estimate(self, datalist):
        """Estimate MS statistics from list of data

        Parameters
        ---------
        datalist : list, shape ('num_data')
            List of several data ([T, dim]) sequence

        Returns
        ---------
        msstats : array, shape (`dim * 2`, `fftsize // 2 + 1`)
            Mean and variance of MS
        """

        # get max length in all data and define fft length
        dim = datalist[0].shape[1]
        maxlen = np.max(list(map(lambda x: x.shape[0], datalist)))
        n_bit = len(bin(maxlen)) - 2
        fftsize = 2 ** (n_bit + 1)

        mss = []
        for data in datalist:
            # create zero padded data
            T = data.shape[0]
            padded_data = np.zeros((fftsize, dim))
            padded_data[:T] += data

            # calculate log power spectum of data
            complex_spec = np.fft.fftn(padded_data, axes=(0, 1))
            log_powerspec = 2 * np.log(np.absolute(complex_spec))
            mss.append(log_powerspec)

        # calculate mean and variance in each freq. bin
        msm = np.mean(np.array(mss), axis=0)
        msv = np.var(np.array(mss), axis=0)
        return np.hstack([msm, msv])

    def postfilter(self, data, msstats, cvmsstats, alpha=1.0, k=0.85, startdim=1):
        """Perform postfilter based on MS statistics to data

        Parameters
        ---------
        data : array, shape (`T`, `dim`)
            Array of data sequence
        msstats : array, shape (`dim * 2`, `fftsize // 2 + 1`)
            Array of mean and variance for target MS
        cvmsstats : array, shape (`dim * 2`, `fftsize // 2 + 1`)
            Array of mean and variance for converted MS
            This option replaces the mean variance of the given data
            into the mean variance estimated from training data.
        alpha : float, optional
            Morphing coefficient between MS transformed data and data.
            .. math::
               alpha * mspf(data) + (1 - alpha) * data
            Default set to 1.0
        k : float, optional
            Postfilter emphasis coefficient [0 <= k <= 1]
            Default set to 1.0
        startdim : int, optional
            Start dimension to perform MS postfilter [1 < startdim < dim]

        Returns
        ---------
        filtered_data : array, shape (`T`, `data`)
            Array of MS postfiltered data sequence
        """

        # get length and dimension
        T, dim = data.shape
        fftsize = msstats.shape[0]
        assert dim == msstats.shape[1] // 2

        # create zero padded data
        padded_data = np.zeros((fftsize, dim))
        padded_data[:T] += data

        # calculate log power spectum of data
        complex_spec = np.fft.fftn(padded_data, axes=(0, 1))
        log_powerspec = 2 * np.log(np.absolute(complex_spec))
        msed_log_powerspec = (1 - k) * log_powerspec + \
            k * ((msstats / cvmsstats)
                 [:, dim:] * (log_powerspec - cvmsstats[:, :dim]) + msstats[:, :dim])

        # reconstruct
        phase_spec = np.angle(complex_spec)
        reconst_complex_spec = np.exp(
            msed_log_powerspec / 2) * (np.cos(phase_spec) + np.sin(phase_spec) * 1j)
        filtered_data = np.fft.ifftn(reconst_complex_spec)[:T]

        if startdim == 1:
            filtered_data[:, 0] = data[:, 0]

        return alpha * filtered_data + (1 - alpha) * data
