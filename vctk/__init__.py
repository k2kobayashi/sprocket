# coding: utf-8

from interface import *


class SpeechParameters(object):

    """
    Speech parameters
    """

    def __init__(self, f0, spectrum_envelope, aperiodicity):
        self.f0 = f0
        self.spectrum_envelope = spectrum_envelope
        self.aperiodicity = aperiodicity


class VoiceConverter(object):

    """
    Voice conversion

    This class assumes:
      - *_parameterizer implements `Parameterizer`
      - *_converter implements `Converter`
      - analyzer implements `Analyzer`
      - synthesizer implments `Synthesizer`

    analyzer and synthesizer must be specified explicitly.

    *_parameterizer and *_converter can be None.

    TODO:
    parameterizerは、デフォでTrasparentParameterizer
    （つまり特徴量をそのままパスするだけのparamterizer）にする？
    """

    def __init__(self,
                 f0_parameterizer=None,
                 f0_converter=None,
                 spectrum_envelope_parameterizer=None,
                 spectrum_envelope_converter=None,
                 aperiodicity_parameterizer=None,
                 aperiodicity_converter=None,
                 analyzer=None,
                 synthesizer=None
                 ):
        self.f0_converter = f0_converter
        self.f0_parameterizer = f0_parameterizer
        self.spectrum_envelope_converter = spectrum_envelope_converter
        self.spectrum_envelope_parameterizer = spectrum_envelope_parameterizer
        self.aperiodicity_converter = aperiodicity_converter
        self.aperiodicity_parameterizer = aperiodicity_parameterizer

        if analyzer == None or synthesizer == None:
            raise "backend must be specified explicitly!"

        self.analyzer = analyzer
        self.synthesizer = synthesizer

        # speech paramters will be stored.
        self.params = None

    def analyze(self, x):
        """
        Decompose speech into parametric representation
        """
        self.params = self.analyzer.analyze(x)

    def convert(self):
        """
        Perform speech parameter conversion
        """
        if self.params == None:
            raise "`analyze` must be called before `convert`"

        if self.f0_converter != None:
            self.params.f0 = self.f0_parameterizer.backward(
                self.f0_converter.convert(
                    self.f0_parameterizer.forward(self.params.f0)
                )
            )

        if self.spectrum_envelope_converter != None:
            self.params.spectrum_envelop = \
                self.spectrum_envelope_parameterizer.backward(
                    self.spectrum_envelope_converter.convert(
                        self.spectrum_envelope_parameterizer.forward(
                            self.params.spectrum_envelope
                        )
                    )
                )

        if self.aperiodicity_converter != None:
            self.params.aperiodicity = self.aperiodicity_parameterizer.backward(
                self.aperiodicity_converter.convert(
                    self.aperiodicity_parameterizer.forward(
                        self.params.aperiodicity)
                )
            )

    def synthesis(self):
        """
        Synthesize speech waveform
        """
        if self.params == None:
            raise "`analyze` must be called before `synthesis`"

        return self.synthesizer.synthesis(self.params)
