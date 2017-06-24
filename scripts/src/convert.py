#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# convert.py
#   First ver.: 2017-06-17
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import os
import argparse
import numpy as np
from scipy.io import wavfile

from sprocket.util.yml import SpeakerYML, PairYML
from sprocket.util.hdf5 import open_h5files, close_h5files
from sprocket.model.GMM import GMMTrainer
from sprocket.stats.gv import GV
from sprocket.stats.f0statistics import F0statistics
from sprocket.backend.synthesizer import Synthesizer
from sprocket.util.jnt import calculate_delta


def main():
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org', type=str,
                        help='original speaker label')
    parser.add_argument('tar', type=str,
                        help='target speaker label')
    parser.add_argument('spkr_ymlf', type=str,
                        help='yml file for the speaker')
    parser.add_argument('pair_ymlf', type=str,
                        help='yml file for the speaker pair')
    args = parser.parse_args()

    # read speaker-and pair-dependent yml file
    sconf = SpeakerYML(args.spkr_ymlf)
    pconf = PairYML(args.pair_ymlf)

    # open eval files
    evh5s = open_h5files(pconf, mode='ev')

    # read F0 transfomer
    orgf0statspath = pconf.pairdir + '/stats/org.f0stats'
    tarf0statspath = pconf.pairdir + '/stats/tar.f0stats'
    f0trans = F0statistics(pconf)
    f0trans.read_statistics(orgf0statspath, tarf0statspath)

    # read GMM for mcep
    mcepgmmpath = pconf.pairdir + '/model/GMM.pkl'
    mcepgmm = GMMTrainer(pconf)
    mcepgmm.open(mcepgmmpath)

    # GV postfilter for mcep
    mcepgvpath = pconf.pairdir + '/stats/tar.gv'
    mcepgv = GV(pconf)
    mcepgv.read_statistics(mcepgvpath)

    # TODO: read GMM for bap

    # open synthesizer
    synthesizer = Synthesizer(sconf)

    # test directory
    testdir = pconf.pairdir + '/test'
    if not os.path.exists(testdir):
        os.makedirs(testdir)

    # file loop
    for h5 in evh5s:
        print(h5.flbl + ' converts.')

        # get F0 feature
        f0 = h5.read('f0')
        mcep = h5.read('mcep')
        mcep_0th = mcep[:, 0]
        apperiodicity = h5.read('ap')

        # convert F0
        cvf0 = f0trans.transform_f0(f0)

        # TODO: convert band-aperiodicity
        # cvbap = bapgmm.convert(calculate_delta(bap))

        # conversion w/o GV
        cvmcep_wopow = mcepgmm.convert(calculate_delta(mcep[:, 1:]))
        cvmcep = np.c_[mcep_0th, cvmcep_wopow]
        # wav = synthesizer.synthesis(f0, cvmcep, apperiodicity)
        wav = synthesizer.synthesis(cvf0, cvmcep, apperiodicity)
        wavpath = testdir + '/' + h5.flbl + '_cv.wav'
        wavfile.write(wavpath, sconf.fs, np.array(wav, dtype=np.int16))

        # conversion w/ GV
        cvmcep_wopow_wGV = mcepgv.gv_postfilter(cvmcep_wopow, sd=1)
        cvmcep_wGV = np.c_[mcep_0th, cvmcep_wopow_wGV]
        wav_wGV = synthesizer.synthesis(cvf0, cvmcep_wGV, apperiodicity)
        wav_wGVpath = testdir + '/' + h5.flbl + '_cv_wGV.wav'
        wavfile.write(wav_wGVpath, sconf.fs, np.array(wav_wGV, dtype=np.int16))

    # close h5 files
    close_h5files(evh5s, 'ev')


if __name__ == '__main__':
    main()
