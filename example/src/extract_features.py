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

import argparse
import numpy as np
from scipy.io import wavfile

from sprocket.feature import FeatureExtractor
from sprocket.util.yml import SpeakerYML
from sprocket.util.hdf5 import HDF5


def main():
    # Options for python
    dcp = 'feature extraction for the speaker'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('speaker', type=str,
                        help='Input speaker label')
    parser.add_argument('ymlf', type=str,
                        help='Yml file of the input speaker')
    parser.add_argument('list_file', type=str,
                        help='List file of the input speaker')
    parser.add_argument('wav_dir', type=str,
                        help='Wav file directory of the speaker')
    parser.add_argument('h5_dir', type=str,
                        help='hdf5 file directory of the speaker')
    args = parser.parse_args()

    # read parameters from yml
    conf = SpeakerYML(args.ymlf)

    # open list file
    with open(args.list_file, 'r') as fp:
        files = fp.readlines()

    for f in files:
        # open wave file
        f = f.rstrip()
        wavf = args.wav_dir + '/' + f + '.wav'
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        assert fs == conf.fs

        print("Processing: " + wavf)
        # constract AcousticFeature clas
        feat = FeatureExtractor(
            x,
            analyzer='world',
            fs=fs,
            minf0=conf.minf0,
            maxf0=conf.maxf0,
        )

        # analyze F0, spc, and ap
        feat.analyze()
        mcep = feat.mcep(dim=conf.dim, alpha=conf.alpha)
        npow = feat.npow()

        # save features into a hdf5 file
        h5f = args.h5_dir + '/' + f + '.h5'
        h5 = HDF5(h5f, mode='w')
        h5.save(feat.spc, ext='spc')
        h5.save(feat.ap, ext='ap')
        h5.save(feat.f0, ext='f0')
        h5.save(mcep, ext='mcep')
        h5.save(npow, ext='npow')
        h5.close()


if __name__ == '__main__':
    main()
