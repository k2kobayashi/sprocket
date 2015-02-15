# coding: utf-8

from vctk import Analyzer, Synthesizer, SpeechParameters
import world


class WORLD(Analyzer, Synthesizer):

    """
    WORLD-based speech analyzer & synthesizer

    TODO:
    support platinum
    """

    def __init__(self,
                 period=5.0,
                 fs=44100,
                 f0_floor=40.0,
                 f0_ceil=700.0,
                 channels_in_octave=2,
                 speed=4
                 ):
        self.period = period
        self.fs = fs
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.channels_in_octave = channels_in_octave
        self.speed = speed

    def analyze(self, x):
        """
        TODO
        """
        opt = world.pyDioOption(self.f0_floor, self.f0_ceil,
                                self.channels_in_octave,
                                self.period, self.speed)

        f0, time_axis = world.dio(x, self.fs, self.period, opt)
        f0 = world.stonemask(x, self.fs, self.period, time_axis, f0)
        spectrum_envelope = world.cheaptrick(x, self.fs, self.period,
                                             time_axis, f0)
        aperiodicity = world.aperiodicityratio(x, self.fs, self.period,
                                               time_axis, f0)
        # TODO
        self.time_len = len(x)

        return SpeechParameters(f0, spectrum_envelope, aperiodicity)

    def synthesis(self, params):
        """
        TODO
        """
        if not isinstance(params, SpeechParameters):
            raise "Not supoprted"

        y = world.synthesis_from_aperiodicity(self.fs, self.period,
                                              params.f0,
                                              params.spectrum_envelope,
                                              params.aperiodicity, self.time_len)
        return y
