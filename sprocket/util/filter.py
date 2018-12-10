# -*- coding: utf-8 -*-

from scipy.signal import firwin, filtfilt


def low_pass_filter(data, cutoff, fs, n_taps=255):
    """Apply low-pass filter

    Parameters
    ----------
    data : array, shape (`T`, `dim`)
        Array of sequence.
    cutoff : int,
        Cutoff frequency
    fs : int,
        Sampling frequency
    n_taps : int, optional
        Tap number

    Returns
    -------
    modified data: array, shape (`T`, `dim`)
        Array of modified sequence.
    """
    if data.shape[0] < n_taps * 3:
        raise ValueError(
            'Length of data should be three times longer than n_taps.')

    fil = firwin(n_taps, cutoff, pass_zero=True, nyq=fs//2)
    modified_data = filtfilt(fil, 1, data, axis=0)
    return modified_data


def high_pass_filter(data, cutoff, fs, n_taps=255):
    """Apply high-pass filter

    Parameters
    ----------
    data : array, shape (`T`, `dim`)
        Array of sequence.
    cutoff : int,
        Cutoff frequency
    fs : int,
        Sampling frequency
    n_taps : int, optional
        Tap number

    Returns
    -------
    modified data: array, shape (`T`, `dim`)
        Array of modified sequence.
    """
    if data.shape[0] < n_taps * 3:
        raise ValueError(
            'Length of data should be three times longer than n_taps.')

    fil = firwin(n_taps, cutoff, pass_zero=False, nyq=fs//2)
    modified_data = filtfilt(fil, 1, data, axis=0)
    return modified_data
