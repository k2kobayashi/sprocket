#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# feature_extraction.py
#   First ver.: 2017-06-05
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
acoustic feature extraction for the speaker

"""

import glob
import argparse
from multiprocessing import Pool

from sprocket.backend import FeatureExtractor
from sprocket.util.yml import SpeakerYML


def feature_analysis(feat, wavf):
    # set wave file
    feat.set_wavf(wavf)

    # extract acoustic features (i.e., F0, spc, ap, mcep, npow)
    feat.analyze_all()

    # save acoustic features as hdf5
    feat.save_hdf5(wavf)

    # read hdf5 files for test
    feat.read_hdf5(wavf)
    return


def fa_wrapper(args):
    return feature_analysis(*args)


def main():
    # Options for python
    description = 'feature extraction for the speaker'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-m', '--multicore', type=int, default=1,
                        help='# of cores for multi-processing')
    parser.add_argument('spkr', type=str,
                        help='input speaker label')
    parser.add_argument('ymlf', type=str,
                        help='configure file for the speaker')
    parser.add_argument('wav_dir', type=str,
                        help='wav file directory of the speaker')
    args = parser.parse_args()

    # read parameters from yml
    conf = SpeakerYML(args.ymlf)

    # construct feature class
    feat = FeatureExtractor(conf)

    # grab .wav files in data directory
    wavs = glob.glob(args.wav_dir + '/' + args.spkr + '/*.wav')

    # feature extraction with WORLD on multi processing
    # feature_analysis(feat, wavs[0])
    p = Pool(args.multicore)
    p.map(fa_wrapper, [(feat, w) for w in wavs])


if __name__ == '__main__':
    main()
