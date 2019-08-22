#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate histograms to decide speaker-dependent parameters

"""

import argparse
import os
import sys

import matplotlib
import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor

matplotlib.use('Agg')  # noqa #isort:skip
import matplotlib.pyplot as plt  # isort:skip


def create_histogram(data, figure_path, range_min=-70, range_max=20,
                     step=10, xlabel='Power [dB]'):
    """Create histogram

    Parameters
    ----------
    data : list,
        List of several data sequences
    figure_path : str,
        Filepath to be output figure
    range_min : int, optional,
        Minimum range for histogram
        Default set to -70
    range_max : int, optional,
        Maximum range for histogram
        Default set to -20
    step : int, optional
        Stap size of label in horizontal axis
        Default set to 10
    xlabel : str, optional
        Label of the horizontal axis
        Default set to 'Power [dB]'

    """

    # plot histgram
    plt.hist(data, bins=200, range=(range_min, range_max),
             density=True, histtype="stepfilled")
    plt.xlabel(xlabel)
    plt.ylabel("Probability")
    plt.xticks(np.arange(range_min, range_max, step))

    figure_dir = os.path.dirname(figure_path)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(figure_path)
    plt.close()


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    dcp = 'Create histogram for speaker-dependent configure'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('speaker', type=str,
                        help='Input speaker label')
    parser.add_argument('list_file', type=str,
                        help='List file of the input speaker')
    parser.add_argument('wav_dir', type=str,
                        help='Directory of wav file')
    parser.add_argument('figure_dir', type=str,
                        help='Directory for figure output')
    args = parser.parse_args(argv)

    # open list file
    with open(args.list_file, 'r') as fp:
        files = fp.readlines()

    f0s = []
    npows = []
    for f in files:
        # open waveform
        f = f.rstrip()
        wavf = os.path.join(args.wav_dir, f + '.wav')
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        print("Extract: " + wavf)

        # constract FeatureExtractor class
        feat = FeatureExtractor(analyzer='world', fs=fs)

        # f0 and npow extraction
        f0, _, _ = feat.analyze(x)
        npow = feat.npow()

        f0s.append(f0)
        npows.append(npow)

    f0s = np.hstack(f0s).flatten()
    npows = np.hstack(npows).flatten()

    # create a histogram to visualize F0 range of the speaker
    f0histogrampath = os.path.join(
        args.figure_dir, args.speaker + '_f0histogram.png')
    create_histogram(f0s, f0histogrampath, range_min=40, range_max=700,
                     step=50, xlabel='Fundamental frequency [Hz]')

    # create a histogram to visualize npow range of the speaker
    npowhistogrampath = os.path.join(
        args.figure_dir, args.speaker + '_npowhistogram.png')
    create_histogram(npows, npowhistogrampath, range_min=-70, range_max=20,
                     step=10, xlabel="Frame power [dB]")


if __name__ == '__main__':
    main()
