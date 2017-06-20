#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# synthesizer.py
#   First ver.: 2017-06-18
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import world
from sprocket.util.yml import SpeakerYML
from sprocket.parameterization import mcgram2spgram


class Synthesizer(object):

    """
    A generic voice synthesizer from acoustic features

    This class assumes:

    TODO:
    Synthesize waveform from F0, ap, and mcep

    Attributes
    ---------
    conf: parameters read from speaker yml file

    """

    def __init__(self, conf):
        # read yml file for the speaker
        self.conf = conf
        return

    def synthesis(self, f0, mcep, aperiodicity):
        """
        synthesis generates waveform from F0, mcep, aperiodicity

        Parameters
        ----------
        f0: array, shape (T, `1`)
          array of F0 sequence
        mcep: array, shape (T, `self.conf.dim`)
          array of mel-cepstrum sequence
        aperiodicity: array, shape (T, `fftlen / 2 + 1`)
          array of aperiodicity

        Return
        ------
        wav: vector, shape (`T`)

        """

        # mcep into spc
        spc = mcgram2spgram(mcep, self.conf.alpha, self.conf.fftl)

        # generate waveform using world vocoder with f0, spc, ap
        x_len = 5 * len(f0) * self.conf.fs / 1000
        wav = world.synthesis_from_aperiodicity(self.conf.fs,
                                                5,
                                                f0,
                                                spc,
                                                aperiodicity,
                                                x_len)

        return wav


def main():
    pass


if __name__ == '__main__':
    main()
