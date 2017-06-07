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
hdf5 feature extraction for the speaker

"""

import os
import glob
import yaml
import h5py
import argparse
import numpy as np
from scipy.io import wavfile
from multiprocessing import Pool


from vctk.backend import WORLD
from vctk.parameterization import spgram2mcgram, spgram2npow


class FeatureExtractor(object):

    """
    wavに紐付いた特徴量を分析し，h5に圧縮して保存するクラス
    - f0, spc, ap, mcep, npow, and more in future (e.g., LPC, cep, some type of mcep, MFCC)
    ToDo: vctkの中に入れる． npowの計算を実装する
    """

    def __init__(self, yml):
        with open(yml) as rf:
            conf = yaml.safe_load(rf)

        self.fs = conf['wav']['fs']
        self.shiftms = conf['wav']['shiftms']
        self.minf0 = conf['f0']['minf0']
        self.maxf0 = conf['f0']['maxf0']
        self.dim = conf['mcep']['dim']
        self.alpha = conf['mcep']['alpha']
        self.analyzer = conf['analyzer']

        if self.analyzer == 'world':
            self.analyzer = WORLD(
                period=self.shiftms, fs=self.fs, f0_floor=self.minf0, f0_ceil=self.maxf0)
        elif self.analyzer == 'SPTK':
            raise('SPTK does not support yet.')
        else:
            raise('other analyzer does not support.')

    def analyze_all(self, wavf):
        # read wav file
        _, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)

        # analysis
        self.features = self.analyzer.analyze(x)
        self.mcep = spgram2mcgram(
            self.features.spectrum_envelope, self.dim, self.alpha)
        self.npow = spgram2npow(self.features.spectrum_envelope)
        return

    def analyze_f0(self, wavf):
        # read wav file
        _, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        return self.analyzer.analyze_f0(x).f0

    def analyze_mcep(self, wavf):
        # read wav file
        _, x = wavfile.read(wavf)
        x = np.array(x, dtype=np.float)
        features = self.analyzer.analyze(x)
        return spgram2mcgram(features.spectrum_envelope, self.dim, self.alpha)

    def save_hdf5(self, wavf):
        h5f = wavf.replace('wav', 'h5')
        dirname, _ = os.path.split(h5f)

        # check directory
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        # check existing h5 file
        if os.path.exists(h5f):
            print("overwrite because HDF5 file already exists. ")

        # write hdf5 file format
        h5 = h5py.File(h5f, "w")
        h5.create_group(dirname)
        h5.create_dataset(
            dirname + '/spc', data=self.features.spectrum_envelope)
        h5.create_dataset(dirname + '/ap', data=self.features.aperiodicity)
        h5.create_dataset(dirname + '/f0', data=self.features.f0)
        h5.create_dataset(dirname + '/mcep', data=self.mcep)
        h5.create_dataset(dirname + '/npow', data=self.npow)
        h5.flush()
        h5.close()

        return


def feature_analysis(feat, wavf):
    # extract all kinds of acoustic features
    feat.analyze_all(wavf)

    # save acoustic features as hdf5
    feat.save_hdf5(wavf)

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

    # construct feature class
    feat = FeatureExtractor(args.ymlf)

    # grab .wav files in data directory
    wavs = glob.glob(args.wav_dir + '/' + args.spkr + '/*.wav')

    # feature extraction with WORLD on multi processing
    p = Pool(args.multicore)
    p.map(fa_wrapper, [(feat, w) for w in wavs])


if __name__ == '__main__':
    main()
