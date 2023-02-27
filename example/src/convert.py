#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Conversion

"""

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
import joblib

from sprocket.model import GV, F0statistics, GMMConvertor
from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5, static_delta

from .misc import low_cut_filter
from .yml import PairYML, SpeakerYML


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-gmmmode', '--gmmmode', type=str, default=None,
                        help='mode of the GMM [None, diff, or intra]')
    parser.add_argument('org', type=str,
                        help='Original speaker')
    parser.add_argument('tar', type=str,
                        help='Target speaker')
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
    args = parser.parse_args(argv)

    # read parameters from speaker yml
    sconf = SpeakerYML(args.org_yml)
    pconf = PairYML(args.pair_yml)

    # read GMM for mcep
    mcepgmmpath = os.path.join(args.pair_dir, 'model/GMM_mcep.pkl')
    mcepgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                           covtype=pconf.GMM_mcep_covtype,
                           gmmmode=args.gmmmode,
                           )
    param = joblib.load(mcepgmmpath)
    mcepgmm.open_from_param(param)
    print("GMM for mcep conversion mode: {}".format(args.gmmmode))

    # read F0 statistics
    stats_dir = os.path.join(args.pair_dir, 'stats')
    orgstatspath = os.path.join(stats_dir,  args.org + '.h5')
    orgstats_h5 = HDF5(orgstatspath, mode='r')
    orgf0stats = orgstats_h5.read(ext='f0stats')
    orgstats_h5.close()

    # read F0 and GV statistics for target
    tarstatspath = os.path.join(stats_dir,  args.tar + '.h5')
    tarstats_h5 = HDF5(tarstatspath, mode='r')
    tarf0stats = tarstats_h5.read(ext='f0stats')
    targvstats = tarstats_h5.read(ext='gv')
    tarstats_h5.close()

    # read GV statistics for converted mcep
    cvgvstatspath = os.path.join(args.pair_dir, 'model', 'cvgv.h5')
    cvgvstats_h5 = HDF5(cvgvstatspath, mode='r')
    cvgvstats = cvgvstats_h5.read(ext='cvgv')
    diffcvgvstats = cvgvstats_h5.read(ext='diffcvgv')
    cvgvstats_h5.close()

    mcepgv = GV()
    f0stats = F0statistics()

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

    # test directory
    test_dir = os.path.join(args.pair_dir, 'test')

    # conversion in each evaluation file
    with open(args.eval_list_file, 'r') as fp:
        for line in fp:
            # open wav file
            f = line.rstrip()
            wavf = os.path.join(args.wav_dir, f + '.wav')
            fs, x = wavfile.read(wavf)
            x = x.astype(np.float64)
            x = low_cut_filter(x, fs, cutoff=70)
            assert fs == sconf.wav_fs

            # create directory
            os.makedirs(os.path.join(test_dir, os.path.dirname(f)), exist_ok=True)

            # analyze F0, mcep, and ap
            f0, spc, ap = feat.analyze(x)
            mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
            mcep_0th = mcep[:, 0]

            # convert F0
            cvf0 = f0stats.convert(f0, orgf0stats, tarf0stats)

            # convert mcep
            cvmcep_wopow = mcepgmm.convert(static_delta(mcep[:, 1:]),
                                           cvtype=pconf.GMM_mcep_cvtype)
            cvmcep = np.c_[mcep_0th, cvmcep_wopow]

            # synthesis VC w/ GV
            if args.gmmmode is None:
                cvmcep_wGV = mcepgv.postfilter(cvmcep,
                                               targvstats,
                                               cvgvstats=cvgvstats,
                                               alpha=pconf.GV_morph_coeff,
                                               startdim=1)
                wav = synthesizer.synthesis(cvf0,
                                            cvmcep_wGV,
                                            ap,
                                            rmcep=mcep,
                                            alpha=sconf.mcep_alpha,
                                            )
                wavpath = os.path.join(test_dir, f + '_VC.wav')

            # synthesis DIFFVC w/ GV
            if args.gmmmode == 'diff':
                cvmcep[:, 0] = 0.0
                cvmcep_wGV = mcepgv.postfilter(mcep + cvmcep,
                                               targvstats,
                                               cvgvstats=diffcvgvstats,
                                               alpha=pconf.GV_morph_coeff,
                                               startdim=1) - mcep
                wav = synthesizer.synthesis_diff(x,
                                                 cvmcep_wGV,
                                                 rmcep=mcep,
                                                 alpha=sconf.mcep_alpha,
                                                 )
                wavpath = os.path.join(test_dir, f + '_DIFFVC.wav')

            # write waveform
            wav = np.clip(wav, -32768, 32767)
            wavfile.write(wavpath, fs, wav.astype(np.int16))
            print(wavpath)


if __name__ == '__main__':
    main()
