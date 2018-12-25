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
        ----------
        datalist : list, shape ('num_data')
            List of several data ([T, dim]) sequence

        Returns
        -------
        msstats : array, shape (`dim * 2`, `fftsize // 2 + 1`)
            Mean and variance of MS
        """

        # get max length in all data and define fft length
        maxlen = np.max(list(map(lambda x: x.shape[0], datalist)))
        n_bit = len(bin(maxlen)) - 2
        self.fftsize = 2 ** (n_bit + 1)

        mss = []
        for data in datalist:
            logpowerspec = self.logpowerspec(data)
            mss.append(logpowerspec)

        # calculate mean and variance in each freq. bin
        msm = np.mean(np.array(mss), axis=0)
        msv = np.var(np.array(mss), axis=0)
        return np.hstack([msm, msv])

    def postfilter(self, data, msstats, cvmsstats, alpha=1.0, k=0.85, startdim=1):
        """Perform postfilter based on MS statistics to data

        Parameters
        ----------
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
        -------
        filtered_data : array, shape (`T`, `data`)
            Array of MS postfiltered data sequence
        """

        # get length and dimension
        T, dim = data.shape
        self.fftsize = msstats.shape[0]
        assert dim == msstats.shape[1] // 2

        # create zero padded data
        logpowerspec, phasespec = self.logpowerspec(data, phase=True)
        msed_logpowerspec = (1 - k) * logpowerspec + \
            k * (np.sqrt((msstats / cvmsstats)[:, dim:]) *
                 (logpowerspec - cvmsstats[:, :dim]) + msstats[:, :dim])

        # reconstruct
        reconst_complexspec = np.exp(msed_logpowerspec / 2) * (np.cos(phasespec) +
                                                               np.sin(phasespec) * 1j)
        filtered_data = np.fft.ifftn(reconst_complexspec)[:T].real

        if startdim == 1:
            filtered_data[:, 0] = data[:, 0]

        # apply morphing
        filtered_data = alpha * filtered_data + (1 - alpha) * data

        return filtered_data

    def logpowerspec(self, data, phase=False):
        """Calculate log power spectrum in each dimension

        Parameters
        ----------
        data : array, shape (`T`, `dim`)
            Array of data sequence

        Returns
        -------
        logpowerspec : array, shape (`dim`, `fftsize // 2 + 1`)
            Log power spectrum of sequence
        phasespec : array, shape (`dim`, `fftsize // 2 + 1`), optional
            Phase spectrum of sequences
        """

        # create zero padded data
        T, dim = data.shape
        padded_data = np.zeros((self.fftsize, dim))
        padded_data[:T] += data

        # calculate log power spectum of data
        complex_spec = np.fft.fftn(padded_data, axes=(0, 1))
        logpowerspec = 2 * np.log(np.absolute(complex_spec))

        if phase:
            phasespec = np.angle(complex_spec)
            return logpowerspec, phasespec
        else:
            return logpowerspec
