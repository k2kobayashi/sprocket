# coding: utf-8
import numpy as np

"""
Interfaces
"""


class Analyzer(object):

    """
    Speech analyzer interface

    All of analyzer must implement this interface.
    """

    def __init__(self):
        pass

    def analyze(self, x):
        """
        Paramters
        ---------
        x: array, shape (`time samples`)
          monoural speech signal in time domain
        """
        raise "Not implemented"


class Synthesizer(object):

    """
    Speech synthesizer interface

    All of synthesizer must implement this interface.
    """

    def __init__(self):
        pass

    def synthesis(self, params):
        """
        Paramters
        ---------
        param: tuple
          speech parameters (f0, spectrum envelop, aperiodicity)
        """
        raise "Not implemented"


class Parameterizer(object):

    """
    Parameterizer interface.

    All parameterizer must implement this interface.
    """

    def __init__(self):
        pass

    def forward(self, raw):
        raise "You must provide a forward parameterization"

    def backward(self, param):
        raise "You must provide s backward parameterization"


class SpectrumEnvelopeParameterizer(Parameterizer):

    """
    Spectrum envelope parameterizer interface

    All spectrum envelope parameterizer must implement this interface.
    """

    def __init__(self):
        pass


class Converter(object):

    """
    Abstract Feature Converter

    All feature converter must implment this interface.
    """

    def __init__(self):
        pass

    def convert(self, feature):
        raise "Not implemented"


class VectorToVectorConverter(Converter):

    """
    Interface of vector-to-vector converter

    All of vector-to-vector converter must implement this class
    """

    def __init__(self):
        pass

    def get_input_shape(self):
        """
        this should return input feature dimention
        """
        raise "Not implemented"

    def get_output_shape(self):
        """
        this should return converted feature dimention
        """
        raise "Not implemented"


class FrameByFrameVectorConverter(VectorToVectorConverter):

    """
    Interface of frame-by-frame vector converter
    """

    def __init__(self):
        pass

    def convert_one_frame(self, feature_vector):
        raise "converters must provide conversion for each time frame"

    def convert(self, feature_matrix):
        """
        Frame-by-frame converters perform conversion for each time frame
        """
        T = len(feature_matrix)
        if self.get_input_shape() != len(feature_matrix[0]):
            raise Exception("Dimention mismatch")

        converted = np.zeros((T, self.get_output_shape()))

        for t in range(T):
            converted[t] = self.convert_one_frame(feature_matrix[t])

        return converted


class MatrixToMatrixConverter(Converter):

    """
    Interface of matrix-to-matrix converter

    All of matrix-to-matrix converter must implement this class
    """

    def __init__(self):
        pass

    def get_input_shape(self):
        """
        this should return input feature dimention
        """
        raise "Not implemented"

    def get_output_shape(self):
        """
        this should return converted feature dimention
        """
        raise "Not implemented"
