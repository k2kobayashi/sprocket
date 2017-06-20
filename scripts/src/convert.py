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

import argparse

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
    f0trans = F0statistics(pconf)

    # read GMM for mcep
    mcepgmmpath = pconf.pairdir + '/model/GMM.pkl'
    mcepgmm = GMMTrainer(pconf)
    mcepgmm.open(mcepgmmpath)

    # TODO: read GMM for bap

    # TODO: GV postfilter for mcep
    # gv = GV(args.pair_ymlf, mode='mcep')

    # open synthesizer
    synthesizer = Synthesizer(sconf)

    # file loop
    for h5 in evh5s:
        print(h5.flbl + ' converts.')

        # get F0 feature
        f0 = h5.read('f0')
        mcep = h5.read('mcep')
        apperiodicity = h5.read('ap')

        # TODO: convert F0
        # cvf0 = f0trans.convert(f0)

        # convert mel-cepstrum
        cvmcep = mcepgmm.convert(calculate_delta(mcep))
        # cvmcep_wGV = gv.postfilter(cvmcep)

        # TODO: convert band-aperiodicity
        # cvbap = bapgmm.convert(calculate_delta(bap))

        # synethesis
        # TODO: need to be bugfix it takes too slow
        wav = synthesizer.synthesis(f0, cvmcep, apperiodicity)
        # wav_wGV = synthesizer.synthesis(cvf0, cvmcep, apperiodicity)

        print 'hogehoge'
        # save as wav file

    # close h5 files
    close_h5files(evh5s)


if __name__ == '__main__':
    main()
