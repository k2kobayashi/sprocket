#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract acoustic features for the speaker

"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor
from sprocket.util import HDF5
from yml import SpeakerYML


def main(*argv):
    argv = argv if argv else sys.argv
    # Options for python
    dcp = 'Extract aoucstic features for the speaker'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('--overwrite', default=False, action='store_true',
                        help='Overwrite h5 file')
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
    args = parser.parse_args(argv)

    # read parameters from speaker yml
    sconf = SpeakerYML(args.ymlf)
    h5_dir = os.path.join(args.pair_dir, 'h5')
    os.makedirs(h5_dir, exist_ok=True)

    # constract FeatureExtractor class
    feat = FeatureExtractor(analyzer=sconf.analyzer,
                            fs=sconf.wav_fs,
                            shiftms=sconf.wav_shiftms,
                            minf0=sconf.f0_minf0,
                            maxf0=sconf.f0_maxf0)

    # open list file
    with open(args.list_file, 'r') as fp:
        for line in fp:
            f = line.rstrip()
            h5f = os.path.join(h5_dir, f + '.h5')

            if (not os.path.exists(h5f)) or args.overwrite:
                wavf = os.path.join(args.wav_dir, f + '.wav')
                fs, x = wavfile.read(wavf)
                x = np.array(x, dtype=np.float)
                assert fs == sconf.wav_fs

                print("Extract acoustic features: " + wavf)

                # analyze F0, spc, ap and bandap
                f0, spc, ap = feat.analyze(x)
                bandap = feat.bandap()
                mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
                npow = feat.npow()

                # save features into a hdf5 file
                h5 = HDF5(h5f, mode='w')
                h5.save(f0, ext='f0')
                h5.save(spc, ext='spc')
                h5.save(ap, ext='ap')
                h5.save(bandap, ext='bandap')
                h5.save(mcep, ext='mcep')
                h5.save(npow, ext='npow')
                h5.close()
            else:
                print("Acoustic features already exist: " + h5f)



if __name__ == '__main__':
    main()
