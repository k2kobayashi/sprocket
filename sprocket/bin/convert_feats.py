#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright (c) 2019 Kazuhiro KOBAYASHI <root.4mac@gmail.com>
#
# Distributed under terms of the MIT license.

"""
Conversion

"""

import argparse
import os
from pathlib import Path
from distutils.util import strtobool

import numpy as np
import joblib
from scipy.io import wavfile

from sprocket.model import GV, F0statistics, GMMConvertor
from sprocket.speech import FeatureExtractor, Synthesizer
from sprocket.util import HDF5, static_delta

from misc import convert_to_continuos_f0, low_cut_filter, low_pass_filter
from yml import PairYML, SpeakerYML


def convert_mcep0th(mcep0th, ostats, tstats):
    """Function to covnert 0th order of mcep

    Parameters
    ----------
    mcep0th : array_like
        0-th order of mel-cepstrum of origianl speaker
    ostats : array_like
        Statistics (i.e., mean and variance) of original speaker
    ostats : array_like
        Statistics (i.e., mean and variance) of target speaker

    Returns
    -------
    cvmcep0th : array_like
        Converted 0-th order of mel-cepstrum

    """
    cvmcep0th = np.sqrt(tstats[1] / ostats[1]) * \
        (mcep0th - ostats[0]) + tstats[0]
    return cvmcep0th


def convert_codeap(codeap, ostats, tstats):
    """Function to transform codeap
    Parameters
    ----------
    codeap : array_like
        codeap of origianl speaker
    ostats : array_like
        Statistics (i.e., mean and variance) of original speaker
    ostats : array_like
        Statistics (i.e., mean and variance) of target speaker
    Returns
    -------
    cvcodeap : array_like
        Converted codeap
    """

    mask = codeap == np.max(codeap)
    cvcodeap = np.sqrt(tstats[1] / ostats[1]) * \
        (codeap - ostats[0]) + tstats[0]
    cvcodeap[mask] = codeap[mask]
    return cvcodeap


def load_stats(statsf):
    """Read speaker-dependent statistics file

    Parameters
    ----------
    statsf : str
        Path for speaker-dependent statistics file (HDF5 format)

    Returns:
    f0stats : array_like
        Statistics for F0
    mcep0thstats : array_like
        Statistics for 0-th order mel-cepstrum
    codeapstats : array_like
        Statisitcs for codeap
    """
    stats_h5 = HDF5(statsf, mode='r')
    f0stats = stats_h5.read(ext='f0stats')
    mcep0thstats = stats_h5.read(ext='mcep0th')
    codeapstats = stats_h5.read(ext='codeap')
    stats_h5.close()

    return f0stats, mcep0thstats, codeapstats


def main():

    # Options for python
    description = 'Convert feature vector and save them for decoding'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--cvmcep0th', default=False, type=strtobool,
                        help='Convert 0-th order of mel-cepstrum')
    parser.add_argument('--cvcodeap', default=False, type=strtobool,
                        help='Convert codeap')
    parser.add_argument('--reanalysis', default=False, type=strtobool,
                        help='Use re-analysis feature of converted voice')
    parser.add_argument('--org_yml', type=str,
                        help='Yml file of original speaker')
    parser.add_argument('--pair_yml', type=str,
                        help='Yml file of speaker pair')
    parser.add_argument('--org_stats', type=str,
                        help='Stats file of source speaker')
    parser.add_argument('--tar_stats', type=str,
                        help='Stats file of target speaker')
    parser.add_argument('--cvgvstats', type=str,
                        help='Stats file of converted feature vector')
    parser.add_argument('--mcepgmmf', type=str,
                        help='GMM for mel-cepstrum')
    parser.add_argument('--iwav', type=str,
                        help='Input wav file')
    parser.add_argument('--owav', type=str,
                        help='Output wav file')
    parser.add_argument('--cvfeats', type=str,
                        help='Converted feature vector file')
    args = parser.parse_args()

    # read parameters from speaker yml
    sconf = SpeakerYML(args.org_yml)
    pconf = PairYML(args.pair_yml)

    # read GMM for mcep
    mcepgmmpath = args.mcepgmmf
    mcepgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                           covtype=pconf.GMM_mcep_covtype,
                           gmmmode=None,
                           )
    param = joblib.load(mcepgmmpath)
    mcepgmm.open_from_param(param)
    print("GMM for mcep conversion mode: {}".format(None))

    # read F0 and mcep0th statistics
    org_f0stats, org_mcep0thstats, org_codeapstats = load_stats(args.org_stats)
    tar_f0stats, tar_mcep0thstats, tar_codeapstats = load_stats(args.tar_stats)

    # read GV statistics
    gvstats_h5 = HDF5(args.tar_stats, mode='r')
    tar_gvstats = gvstats_h5.read(ext='gv')
    gvstats_h5.close()

    cvgvstats_h5 = HDF5(args.cvgvstats, mode='r')
    cvgvstats = cvgvstats_h5.read(ext='cvgv')
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

    # conversion in each evaluation file
    # open wav file
    wavf = Path(args.iwav)
    fs, x = wavfile.read(str(wavf))
    x = x.astype(np.float)
    x = low_cut_filter(x, fs, cutoff=70)
    assert fs == sconf.wav_fs

    # analyze F0, mcep, and codeap
    f0, _, _ = feat.analyze(x)
    mcep = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
    mcep0th = mcep[:, 0]
    codeap = feat.codeap()

    # convert F0
    cvf0 = f0stats.convert(f0, org_f0stats, tar_f0stats)
    uv, cvcf0 = convert_to_continuos_f0(cvf0)
    cvcf0 = low_pass_filter(cvcf0, int(1.0 / (sconf.shiftms * 0.001)), cutoff=20)

    # convert mcep
    cvmcep_wopow = mcepgmm.convert(static_delta(mcep[:, 1:]),
                                   cvtype=pconf.GMM_mcep_cvtype)
    if args.cvmcep0th:
        cvmcep0th = convert_mcep0th(mcep0th, org_mcep0thstats,
                                    tar_mcep0thstats)
    else:
        cvmcep0th = mcep0th
    cvmcep = np.c_[cvmcep0th, cvmcep_wopow]
    cvmcep_wGV = mcepgv.postfilter(cvmcep,
                                   tar_gvstats,
                                   cvgvstats=cvgvstats,
                                   alpha=pconf.GV_morph_coeff,
                                   startdim=1)

    # convert codeap
    if args.cvcodeap:
        cvcodeap = convert_codeap(codeap,
                                  org_codeapstats, tar_codeapstats)
    else:
        cvcodeap = codeap

    # synthesis VC w/ GV
    wav = synthesizer.synthesis(cvf0,
                                cvmcep_wGV,
                                cvcodeap,
                                rmcep=mcep,
                                alpha=sconf.mcep_alpha,
                                )
    wav = np.clip(wav, -32768, 32767)
    wavfile.write(args.owav, sconf.wav_fs, wav.astype(np.int16))

    if args.reanalysis:
        # reanalysis converted waveform
        f0rate = np.exp((tar_f0stats[1] / org_f0stats[1]))
        feat = FeatureExtractor(analyzer=sconf.analyzer,
                                fs=sconf.wav_fs,
                                fftl=sconf.wav_fftl,
                                shiftms=sconf.wav_shiftms,
                                minf0=sconf.f0_minf0 * f0rate,
                                maxf0=sconf.f0_maxf0 * f0rate)
        cvf0, _, _ = feat.analyze(wav)
        cvmcep_wGV = feat.mcep(dim=sconf.mcep_dim, alpha=sconf.mcep_alpha)
        cvcodeap = feat.codeap()
        uv, cvcf0 = convert_to_continuos_f0(cvf0)
        cvcf0 = low_pass_filter(cvcf0, int(1.0 / (sconf.shiftms * 0.001)), cutoff=20)
        wav = synthesizer.synthesis(cvf0,
                                    cvmcep_wGV,
                                    cvcodeap,
                                    alpha=sconf.mcep_alpha,
                                    )
        wav = np.clip(wav, -32768, 32767)
        wavfile.write(args.owav + 'reanalysis.wav',
                      sconf.wav_fs, wav.astype(np.int16))

    # save hdf5 file for decode
    h5 = HDF5(args.cvfeats, 'w')
    cvfeatures = np.c_[uv, cvcf0, cvmcep_wGV, cvcodeap]
    h5.save(cvfeatures, 'world')
    h5.close()


if __name__ == '__main__':
    main()
