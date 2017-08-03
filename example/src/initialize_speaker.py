#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# init_spkr.py
#   First ver.: 2017-06-05
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
Generate F0 histgram

"""

import os
import argparse
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sprocket.feature import FeatureExtractor


def create_f0_histogram(f0s, f0histogrampath):
    # plot histgram
    plt.hist(f0s, bins=200, range=(40, 700),
             normed=True, histtype="stepfilled")
    plt.xlabel("Fundamental frequency")
    plt.ylabel("Probability")

    figure_dir = os.path.dirname(f0histogrampath)
    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    plt.savefig(f0histogrampath)


def main():
    dcp = 'create speaker-dependent configure file (speaker.yml)'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('speaker', type=str,
                        help='Input speaker label')
    parser.add_argument('list_file', type=str,
                        help='List file of the input speaker')
    parser.add_argument('wav_dir', type=str,
                        help='Directory of wav file')
    parser.add_argument('pair_dir', type=str,
                        help='Directory of speaker pair')
    args = parser.parse_args()

    # open list file
    with open(args.list_file, 'r') as fp:
        files = fp.readlines()

    f0s = []
    for f in files:
        # open waveform
        f = f.rstrip()
        wavf = os.path.join(args.wav_dir, f + '.wav')
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        print("Extract f0: " + wavf)

        # constract FeatureExtractor clas
        feat = FeatureExtractor(x, analyzer='world', fs=fs)
        feat.analyze()

        # f0 extraction
        f0s.append(feat.f0())

    # create a figure to visualize F0 range of the speaker
    figure_dir = os.path.join(args.pair_dir, 'figure')
    f0histogrampath = os.path.join(
        figure_dir, args.speaker + '_f0histogram.png')
    create_f0_histogram(np.hstack(f0s).flatten(), f0histogrampath)

if __name__ == '__main__':
    main()
