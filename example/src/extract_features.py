#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract acoustic features for the speaker

"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5

from .misc import low_cut_filter
from .yml import SpeakerYML


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
    anasyn_dir = os.path.join(args.pair_dir, 'anasyn')
    if not os.path.exists(os.path.join(h5_dir, args.speaker)):
        os.makedirs(os.path.join(h5_dir, args.speaker))
    if not os.path.exists(os.path.join(anasyn_dir, args.speaker)):
        os.makedirs(os.path.join(anasyn_dir, args.speaker))

    # constract FeatureExtractor class
    feat = FeatureExtractor(analyzer=sconf.analyzer,
                            fs=sconf.wav_fs,
                            fftl=sconf.wav_fftl,
                            shiftms=sconf.wav_shiftms,
                            minf0=sconf.f0_minf0,
                            maxf0=sconf.f0_maxf0)

    # constract Synthesizer class
    synthesizer = Synthesizer(fs=sconf.wav_fs,
                              fftl=sconf.wav_fftl,
                              shiftms=sconf.wav_shiftms)

    # open list file
    with open(args.list_file, 'r') as fp:
        for line in fp:
            f = line.rstrip()
            h5f = os.path.join(h5_dir, f + '.h5')

            if (not os.path.exists(h5f)) or args.overwrite:
                wavf = os.path.join(args.wav_dir, f + '.wav')
                fs, x = wavfile.read(wavf)
                x = np.array(x, dtype=np.float)
                x = low_cut_filter(x, fs, cutoff=70)
                assert fs == sconf.wav_fs

                print("Extract acoustic features: " + wavf)

                # analyze F0, spc, and ap
                f0, spc, ap = feat.analyze(x)
                mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
                npow = feat.npow()
                codeap = feat.codeap()

                # save features into a hdf5 file
                h5 = HDF5(h5f, mode='w')
                h5.save(f0, ext='f0')
                # h5.save(spc, ext='spc')
                # h5.save(ap, ext='ap')
                h5.save(mcep, ext='mcep')
                h5.save(npow, ext='npow')
                h5.save(codeap, ext='codeap')
                h5.close()

                # analysis/synthesis using F0, mcep, and ap
                wav = synthesizer.synthesis(f0,
                                            mcep,
                                            ap,
                                            alpha=sconf.mcep_alpha,
                                            )
                wav = np.clip(wav, -32768, 32767)
                anasynf = os.path.join(anasyn_dir, f + '.wav')
                wavfile.write(anasynf, fs, np.array(wav, dtype=np.int16))
            else:
                print("Acoustic features already exist: " + h5f)


if __name__ == '__main__':
    main()
