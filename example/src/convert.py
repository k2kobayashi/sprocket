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

from __future__ import division, print_function, absolute_import

import os
import argparse
import numpy as np
from scipy.io import wavfile
import pysptk
from pysptk.synthesis import MLSADF

from sprocket.util.hdf5 import HDF5files
from sprocket.model.GMM import GMMConvertor
from sprocket.stats.gv import GV
from sprocket.stats.f0statistics import F0statistics
from sprocket.feature.synthesizer import Synthesizer
from sprocket.util.delta import delta

from yml import SpeakerYML, PairYML


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-gmmmode', '--gmmmode', type=str, default=None,
                        help='mode of the GMM [None, diff, or intra]')
    parser.add_argument('org', type=str,
                        help='Original speaker')
    parser.add_argument('tar', type=str,
                        help='Original speaker')
    parser.add_argument('org_yml', type=str,
                        help='Yml file of the original speaker')
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('eval_list_file', type=str,
                        help='List file for evaluation')
    parser.add_argument('wav_dir', type=str,
                        help='Directory path of source spekaer')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of pair directory')
    args = parser.parse_args()

    # read parameters from speaker yml
    sconf = SpeakerYML(args.org_yml)
    pconf = PairYML(args.pair_yml)

    # open evaluation files from list
    h5_dir = os.path.join(args.pair_dir, 'h5')
    eval_h5s = HDF5files(args.eval_list_file, h5_dir)

    # read F0 statistics file
    orgf0statspath = os.path.join(
        args.pair_dir, 'stats', args.org + '.f0stats')
    tarf0statspath = os.path.join(
        args.pair_dir, 'stats', args.tar + '.f0stats')
    f0stats = F0statistics()
    f0stats.open_from_file(orgf0statspath, tarf0statspath)

    # read GMM for mcep
    mcepgmmpath = os.path.join(args.pair_dir, 'model/GMM.pkl')
    mcepgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix, covtype=pconf.GMM_mcep_covtype,
                           gmmmode=args.gmmmode, cvtype=pconf.GMM_mcep_cvtype)
    mcepgmm.open(mcepgmmpath)
    print("conversion mode: {}".format(args.gmmmode))

    # GV postfilter for mcep
    mcepgvpath = os.path.join(args.pair_dir, 'stats', args.tar + '.gv')
    mcepgv = GV()
    mcepgv.open_from_file(mcepgvpath)

    # open synthesizer
    synthesizer = Synthesizer()
    alpha = pysptk.util.mcepalpha(sconf.wav_fs)
    shiftl = int(sconf.wav_fs / 1000 * sconf.wav_shiftms)
    mlsa_fil = pysptk.synthesis.Synthesizer(
        MLSADF(order=sconf.mcep_dim, alpha=sconf.mcep_alpha), shiftl)

    # test directory
    testdir = os.path.join(args.pair_dir, 'test')
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    # conversion in each evaluation file
    for h5 in eval_h5s.h5list:
        wavpath = os.path.join(args.wav_dir, args.org,
                               "{}.wav".format(h5.flbl))
        assert os.path.exists(wavpath)
        fs, x = wavfile.read(wavpath)
        print('convert ' + h5.flbl)

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

        # synthesis VC w/ GV
        if args.gmmmode == None:
            cvmcep_wGV = mcepgv.postfilter(cvmcep, startdim=1)
            wav = synthesizer.synthesis(cvf0, cvmcep_wGV, apperiodicity,
                                        alpha=sconf.mcep_alpha, fftl=sconf.wav_fftl,
                                        fs=sconf.wav_fs)

            wav = np.clip(wav, -32768, 32767)
            wavpath = os.path.join(testdir, h5.flbl + '_VC.wav')

        # synthesis DIFFVC w/ GV
        if args.gmmmode == 'diff':
            cvmcep[:, 0] = 0.0
            cvmcep_wGV = mcepgv.postfilter(mcep + cvmcep, startdim=1) - mcep
            b = np.apply_along_axis(pysptk.mc2b, 1, cvmcep_wGV, alpha)
            assert np.isfinite(b).all()
            x = x.astype(np.float64)
            wav = mlsa_fil.synthesis(x, b)
            wav = np.clip(wav, -32768, 32767)
            wavpath = os.path.join(testdir, h5.flbl + '_DIFFVC.wav')

        # write waveform
        wavfile.write(
            wavpath, fs, np.array(wav, dtype=np.int16))

    # close h5 files
    eval_h5s.close()


if __name__ == '__main__':
    main()
