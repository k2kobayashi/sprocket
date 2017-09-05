#! /usr/bin/env python
# -*- coding: utf-8 -*-
#
# F0_transformation.py
#   First ver.: 2017/07/20
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
Transform F0 of original waveform

"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

from sprocket.feature import FeatureExtractor
from sprocket.stats.f0statistics import F0statistics
from sprocket.util.shifter import Shifter
from yml import SpeakerYML


def get_f0s_from_list(conf, list_file, wav_dir):
    # open list file
    with open(list_file, 'r') as fp:
        files = fp.readlines()

    f0s = []
    for f in files:
        # open wave file
        f = f.rstrip()
        wavf = os.path.join(wav_dir, f + '.wav')
        fs, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        assert fs == conf.wav_fs

        print("Extract F0: " + wavf)
        # constract FeatureExtractor clas
        feat = FeatureExtractor(x, analyzer=conf.analyzer, fs=conf.wav_fs,
                                shiftms=conf.wav_shiftms,
                                minf0=conf.f0_minf0, maxf0=conf.f0_maxf0)
        feat.analyze()
        f0s.append(feat.f0())

    return f0s


def transform_f0_from_list(f0rate, wav_fs, list_file, wav_dir):
    # open list file
    with open(list_file, 'r') as fp:
        files = fp.readlines()

    # Construct Shifter class
    shifter = Shifter(wav_fs, f0rate=f0rate)

    # check output directory
    transformed_wavdir = os.path.join(wav_dir + '_' + str(f0rate))
    if not os.path.exists(transformed_wavdir):
        os.makedirs(transformed_wavdir)

    for f in files:
        # open wave file
        f = f.rstrip()
        wavf = os.path.join(wav_dir, f + '.wav')

        # output file path
        transformed_wavpath = os.path.join(
            transformed_wavdir, os.path.basename(wavf))

        if not os.path.exists(transformed_wavpath):
            # transform F0 of waveform
            fs, x = wavfile.read(wavf)
            x = np.array(x, dtype=np.float)
            assert fs == wav_fs
            transformed_x = shifter.f0transform(x)

            wavfile.write(transformed_wavpath, fs,
                          transformed_x.astype(np.int16))
            print('F0 transformed wav file: ' + transformed_wavpath)
        else:
            print('F0 transformed wav file already exists: ' + transformed_wavpath)


def create_F0_transformed_list_file(speaker, f0rate, list_file):
    # open list file
    with open(list_file, 'r') as fp:
        files = fp.readlines()

    transformed_speaker = speaker + '_' + str(f0rate)

    transformed_list = []
    for f in files:
        transformed_list.append(f.replace(speaker, transformed_speaker))

    # write F0 transformed list file
    listpath = list_file.replace(speaker, transformed_speaker)
    with open(listpath, 'w') as fp:
        fp.writelines(transformed_list)


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    dcp = 'Extract aoucstic features for the speaker'
    parser = argparse.ArgumentParser(description=dcp)
    parser.add_argument('speaker', type=str,
                        help='Original speaker label')
    parser.add_argument('org_yml', type=str,
                        help='Yml file of the original speaker')
    parser.add_argument('tar_yml', type=str,
                        help='Yml file of the target speaker')
    parser.add_argument('org_train_list', type=str,
                        help='List file of the original speaker')
    parser.add_argument('org_eval_list', type=str,
                        help='List file of the original speaker')
    parser.add_argument('tar_train_list', type=str,
                        help='List file of the target speaker')
    parser.add_argument('wav_dir', type=str,
                        help='Wav file directory of the speaker')
    args = parser.parse_args(argv)

    # read parameters from speaker yml
    org_conf = SpeakerYML(args.org_yml)
    tar_conf = SpeakerYML(args.tar_yml)

    # get f0 list to calculate F0 transformation ratio
    org_f0s = get_f0s_from_list(org_conf, args.org_train_list, args.wav_dir)
    tar_f0s = get_f0s_from_list(tar_conf, args.tar_train_list, args.wav_dir)

    # calculate F0 statistics of original and target speaker
    f0stats = F0statistics()
    f0stats.estimate(org_f0s)
    org_f0stats = f0stats.f0stats
    f0stats.estimate(tar_f0s)
    tar_f0stats = f0stats.f0stats

    # calculate F0 transformation ratio between original and target speakers
    f0rate = np.round(np.exp(tar_f0stats[0] - org_f0stats[0]), decimals=2)
    print('F0 transformation ratio: ' + str(f0rate))

    # F0 transformation of original waveform in both train and eval list files
    transform_f0_from_list(
        f0rate, org_conf.wav_fs, args.org_train_list, args.wav_dir)
    transform_f0_from_list(
        f0rate, org_conf.wav_fs, args.org_eval_list, args.wav_dir)

    # create list files for F0 transformed original
    create_F0_transformed_list_file(args.speaker, f0rate, args.org_train_list)
    create_F0_transformed_list_file(args.speaker, f0rate, args.org_eval_list)


if __name__ == '__main__':
    main()
