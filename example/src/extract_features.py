#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# extract_features.py
#   First ver.: 2017-06-05
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
Extract acoustic features for the speaker

"""

import os
import argparse
import numpy as np
from scipy.io import wavfile

from sprocket.feature import FeatureExtractor
from sprocket.util.hdf5 import HDF5

from yml import SpeakerYML


def main():
    # Options for python
    dcp = 'Extract aoucstic features for the speaker'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('speaker', type=str,
                        help='Input speaker label')
    parser.add_argument('ymlf', type=str,
                        help='Yml file of the input speaker')
    parser.add_argument('list_file', type=str,
                        help='List file of the input speaker')
    parser.add_argument('wav_dir', type=str,
                        help='Wav file directory of the speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Directory of the speaker pair')
    args = parser.parse_args()

    # read parameters from speaker yml
    sconf = SpeakerYML(args.ymlf)

    # open list file
    with open(args.list_file, 'r') as fp:
        files = fp.readlines()

    for f in files:
        # open wave file
        f = f.rstrip()
        wavf = os.path.join(args.wav_dir, f + '.wav')
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        assert fs == sconf.wav_fs

        print("Extract acoustic features: " + wavf)
        # constract FeatureExtractor clas
        feat = FeatureExtractor(x, analyzer=sconf.analyzer, fs=sconf.wav_fs,
                                shiftms=sconf.wav_shiftms,
                                minf0=sconf.f0_minf0, maxf0=sconf.f0_maxf0)

        # analyze F0, spc, and ap
        feat.analyze()
        f0 = feat.f0()
        spc = feat.spc()
        ap = feat.ap()
        mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
        npow = feat.npow()

        # save features into a hdf5 file
        h5_dir = os.path.join(args.pair_dir, 'h5')
        if not os.path.exists(h5_dir):
            os.makedirs(h5_dir)
        h5f = os.path.join(h5_dir + '/' + f + '.h5')
        h5 = HDF5(h5f, mode='w')
        h5.save(f0, ext='f0')
        h5.save(spc, ext='spc')
        h5.save(ap, ext='ap')
        h5.save(mcep, ext='mcep')
        h5.save(npow, ext='npow')
        h5.close()

if __name__ == '__main__':
    main()
