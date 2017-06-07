# coding: utf-8
import numpy as np

"""
Interfaces
"""


class Analyzer(object):

    """
    Speech analyzer interface. `Analyzer` should convert a time-domain speech
    signal into a set of speech parameters (F0, spectrum envelope and
    aperiodicity, etc).

    All analyzer must implement this interface.
    """

    def __init__(self):
        pass

    def analyze(self, x):
        """
        analyze decomposes a speech signal `x` (in time-domain) into
        a set of speech parameters:

        Paramters
        ---------
        x: array, shape (`time samples`)
          monoural speech signal in time domain
        """
        raise NotImplementedError("")


class Synthesizer(object):

    """
    Speech synthesizer interface. `Synthesizer` should generate speech `waveform`
    from a set of speech parameters.

    All synthesizer must implement this interface.
    """

    def __init__(self):
        pass

    def synthesis(self, params):
        """
        sythesis re-synthesizes a speech waveform from speech paramters.

        Paramters
        ---------
        param: tuple
          speech parameters (f0, spectrum envelop, aperiodicity)
        """
        raise NotImplementedError("")


class ForwardParameterizer(object):

    """
    Forward parameterizer interface.

    All bi-direct parameterizer must implement this interface.
    """

    def __init__(self):
        pass

    def forward(self, raw):
        """
        forward converts raw speech feature to (lower-dimentional) speech feature

        e.g. spectrum envelope -> mel-cepstrum

        Parameters
        ----------
        raw:
          raw speech feature
        """
        raise NotImplementedError("You must provide a forward parameterization")


class BackwardParameterizer(object):

    def __init__(self):
        pass

    def backward(self, param):
        """
        backward reconstructs raw speech feature (that can be used directly in
        speech waveform synthesis) from parameterized speech feature.

        e.g. mel-cepstrum -> spectrum envelope

        Parameters
        ----------
        param:
          parameterized speech feature
        """
        raise NotImplementedError(
            "You must provide s backward parameterization")


class BidirectParameterizer(ForwardParameterizer, BackwardParameterizer):

    """
    Bi-directional parameterizer interface. `BidirectParameterizer` should
    provide bi-directional conversion between raw speech parameters (such as
    spectrum envelope) and low-dimentional speech features (such as
    mel-cesptrum).

    All bi-directional parameterizer must implement this interface.
    """

    def __init__(self):
        super(BidirectParameterizer, self).__init__()


class Converter(object):

    """
    Abstract Converter.

    All converter must implment this interface.
    """

    def __init__(self):
        pass

    def get_input_shape(self):
        """
        this should return input feature dimension
        """
        raise NotImplementedError("")

    def get_output_shape(self):
        """
        this should return converted feature dimension
        """
        raise NotImplementedError("")

    def convert(self, feature):
        """
        convert input feature


        Parameters
        ----------
        feature:
          input feature to be converted
        """
        raise NotImplementedError("")


class FrameByFrameVectorConverter(Converter):

    """
    Interface of frame-by-frame vector converter
    """

    def __init__(self):
        pass

    def convert_one_frame(self, feature_vector):
        """
        convert input feature vector

        Paramters
        ---------
        feature_vector: array

        """
        raise NotImplementedError(
            "converters must provide conversion for each time frame")

    def convert(self, feature_matrix):
        """
        Frame-by-frame converters perform conversion for each time frame

        Parameters
        ----------
        feature_matrix: array

        """
        T = len(feature_matrix)
        if self.get_input_shape() != len(feature_matrix[0]):
            raise Exception("Dimention mismatch")

        converted = np.zeros((T, self.get_output_shape()))

        for t in range(T):
            converted[t] = self.convert_one_frame(feature_matrix[t])

        return converted
