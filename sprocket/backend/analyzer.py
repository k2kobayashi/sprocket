#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# analyzer.py
#   First ver.: 2017-06-09
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""
WORLD-based speech analyzer & synthesizer

"""

import world


class WORLD(object):

    """
    WORLD-based speech analyzer & synthesizer

    TODO:
    support platinum

    Attributes
    ----------
    period: float
      frame period (default: 5.0)
    fs: int
      sampling frequency (default: 44100)
    f0_floor: float
      floor in f0 estimation
    f0_ceil: float
      ceil in f0 estimation
    channels_in_octave: int
      number of F0 search candidates for each octave
    speed: int
      re-sampling parameter (see WORLD for details)
    time_len: int
      time length for analyzed speech signal
    """

    def __init__(self,
                 period=5.0,
                 fs=44100,
                 f0_floor=40.0,
                 f0_ceil=700.0,
                 channels_in_octave=2,
                 speed=4
                 ):
        super(WORLD, self).__init__()

        self.period = period
        self.fs = fs
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.channels_in_octave = channels_in_octave
        self.speed = speed

        self.opt = world.pyDioOption(self.f0_floor, self.f0_ceil,
                                     self.channels_in_octave,
                                     self.period, self.speed)

    def analyze(self, x):
        """
        analyze decomposes a speech signal into three parameters:
          1. Fundamental frequency
          2. Spectrum envelope
          3. Aperiodicity ratio

        Paramters
        ---------
        x: array, shape (`time samples`)
          monoral speech signal in time domain
        """
        f0, time_axis = world.dio(x, self.fs, self.period, self.opt)
        f0 = world.stonemask(x, self.fs, self.period, time_axis, f0)
        spectrum_envelope = world.cheaptrick(x, self.fs, self.period,
                                             time_axis, f0)
        aperiodicity = world.aperiodicityratio(x, self.fs, self.period,
                                               time_axis, f0)
        # TODO
        self.time_len = len(x)

        return f0, spectrum_envelope, aperiodicity

    def analyze_f0(self, x):
        """
        analyze decomposes a speech signal into F0:

        Paramters
        ---------
        x: array, shape (`time samples`)
          monoral speech signal in time domain
        """
        f0, time_axis = world.dio(x, self.fs, self.period, self.opt)
        f0 = world.stonemask(x, self.fs, self.period, time_axis, f0)

        return f0

    def synthesis(self, f0, spc, ap):
        """
        synthesis re-synthesizes a speech waveform from:
          1. Fundamental frequency
          2. Spectrum envelope
          3. Aperiodicity ratio

        Parameters
        ----------
        params: SpeechParamaeters
          a set of speech parameters

        """
        y = world.synthesizey(f0, spc, ap, self.fs, 5)
        return y
