# coding: utf-8

from interface import *

from vctk.parameterization import TransparentParameterizer


class SpeechParameters(object):

    """
    Speech parameters

    TODO:
    how to handle other paramters?

    Attributes
    ----------

    f0: array, shape (`T`)
      fundamental frequency
    spectrum_envelope: array, shape (`T`, `fftlen/2+1`)
      spectrum envelope
    aperiodicity: array, shape  (`T`, `fftlen/2+1`)
      aperiodicity spectrum

    """

    def __init__(self, f0, spectrum_envelope, aperiodicity):
        if spectrum_envelope is not None and aperiodicity is not None:
            assert len(f0) == len(spectrum_envelope) == len(aperiodicity)

        self.f0 = f0
        self.spectrum_envelope = spectrum_envelope
        self.aperiodicity = aperiodicity


class VoiceConverter(object):

    """
    A generic voice converter

    This class assumes:
      - *_parameterizer implements `BidirectParameterizer`
      - *_converter implements `Converter`
      - analyzer implements `Analyzer`
      - synthesizer implments `Synthesizer`

    analyzer and synthesizer must be specified explicitly.

    *_parameterizer and *_converter can be None.

    TODO:
    rename (VocoderBasedVoiceConverter?)

    Attributes
    ---------
    f0_parameterizer: `BidirectParameterizer`
      parameterizer for fundamental frequency (e.g. LogarithmicParameterizer)
    f0_converter: `Converter`
      converter for f0
    spectrum_envelope_parameterizer: `BidirectParameterizer`
      parameterizer for spectrum envelope (e.g. MelCepstrumParameterizer)
    spectrum_envelope_converter: `Converter`
      converter for spectrum enveloep (e.g. JointGMMConverter)
    aperiodicity_parameterizer: `BidirectParameterizer`
      parameterizer for aperiodicity (e.g. TODO)
    aperiodicity_converter: `Converter`
      converter for aperiodicity (e.g. TODO)
    analyzer: `Analyzer`
      speech analyze engine
    synthesizer: `Synthesizer`
      speech synthesis engine
    """

    def __init__(self,
                 f0_parameterizer=TransparentParameterizer(),
                 f0_converter=None,
                 spectrum_envelope_parameterizer=TransparentParameterizer(),
                 spectrum_envelope_converter=None,
                 aperiodicity_parameterizer=TransparentParameterizer(),
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
            raise RuntimeError("backend must be specified explicitly!")

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
            raise RuntimeError("`analyze` must be called before `convert`")

        if self.f0_converter != None:
            self.params.f0 = self.f0_parameterizer.backward(
                self.f0_converter.convert(
                    self.f0_parameterizer.forward(self.params.f0)
                )
            )

        if self.spectrum_envelope_converter != None:
            self.params.spectrum_envelope = \
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
            raise RuntimeError("`analyze` must be called before `synthesis`")

        return self.synthesizer.synthesis(self.params)
