#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# convert.py
#   First ver.: 2017-06-17
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp> #
#   Distributed under terms of the MIT license.
#

"""


"""

import os
import argparse
import numpy as np
from scipy.io import wavfile

from sprocket.util.hdf5 import HDF5files
from sprocket.model.GMM import GMMConvertor
from sprocket.stats.gv import GV
from sprocket.stats.f0statistics import F0statistics
from sprocket.feature.synthesizer import Synthesizer
from sprocket.util.delta import delta

import pysptk
from pysptk.synthesis import MLSADF
from os.path import join, exists


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-cvtype', '--cvtype', type=str, default=None,
                        help='type of the conversion [None, diff, or intra]')
    parser.add_argument('org', type=str,
                        help='Original speaker')
    parser.add_argument('tar', type=str,
                        help='Original speaker')
    parser.add_argument('evlistf', type=str,
                        help='List file for evaluation')
    parser.add_argument('wav_dir', type=str,
                        help='Directory path of source spekaer')
    parser.add_argument('h5_dir', type=str,
                        help='Directory path of hdf5 files')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of pair directory')
    args = parser.parse_args()

    # open evaluation files from list
    evh5s = HDF5files(args.evlistf, args.h5_dir)

    # read F0 transfomer
    orgf0statspath = args.pair_dir + '/stats/' + args.org + '.f0stats'
    tarf0statspath = args.pair_dir + '/stats/' + args.tar + '.f0stats'
    f0stats = F0statistics()
    f0stats.open_from_file(orgf0statspath, tarf0statspath)

    # read GMM for mcep
    mcepgmmpath = args.pair_dir + '/model/GMM.pkl'
    mcepgmm = GMMConvertor(n_mix=32, covtype='full',
                           gmmmode=args.cvtype, cvtype='mlpg')
    mcepgmm.open(mcepgmmpath)
    print("mode: {}".format(args.cvtype))

    # GV postfilter for mcep
    mcepgvpath = args.pair_dir + '/stats/' + args.tar + '.gv'
    mcepgv = GV()
    mcepgv.open_from_file(mcepgvpath)

    # open synthesizer
    synthesizer = Synthesizer()
    alpha = pysptk.util.mcepalpha(16000)
    diff_synth = pysptk.synthesis.Synthesizer(
        MLSADF(order=24, alpha=alpha), 80)

    # test directory
    testdir = args.pair_dir + '/test'
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    # file loop
    for h5 in evh5s.h5list[:5]:
        src_wavpath = join(args.wav_dir, args.org,
                           "{}.wav".format(h5.flbl))
        assert exists(src_wavpath)
        fs, src_waveform = wavfile.read(src_wavpath)
        print(h5.flbl + ' converts.')

        # get F0 feature
        f0 = h5.read('f0')
        mcep = h5.read('mcep')
        mcep_0th = mcep[:, 0]
        apperiodicity = h5.read('ap')

        # convert F0
        cvf0 = f0stats.convert(f0)

        # convert mel-cepstrum
        cvmcep_wopow = mcepgmm.convert(np.c_[mcep[:, 1:], delta(mcep[:, 1:])])
        cvmcep = np.c_[mcep_0th, cvmcep_wopow]

        if args.cvtype == None:
            # synthesis w/ GV
            cvmcep_wGV = mcepgv.postfilter(cvmcep, startdim=1)
            wav = synthesizer.synthesis(cvf0, cvmcep_wGV, apperiodicity)
            wav = np.clip(wav, -32768, 32767)
            wavpath = testdir + '/' + h5.flbl + '_VC.wav'

        if args.cvtype == 'diff':
            # remove power coef
            cvmcep[:, 0] = 0.0
            cvmcep_wGV = mcepgv.postfilter(mcep + cvmcep, startdim=1) - mcep
            b = np.apply_along_axis(pysptk.mc2b, 1, cvmcep_wGV, alpha)
            assert np.isfinite(b).all()
            src_waveform = src_waveform.astype(np.float64)
            wav = diff_synth.synthesis(src_waveform, b)
            wav = np.clip(wav, -32768, 32767)

            wavpath = testdir + '/' + h5.flbl + '_DIFFVC.wav'

        wavfile.write(
            wavpath, fs, np.array(wav, dtype=np.int16))

    # close h5 files
    evh5s.close()


if __name__ == '__main__':
    main()
