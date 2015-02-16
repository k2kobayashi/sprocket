from gmmmap import *

from vctk import Converter


class Multiplier(Converter):

    """
    Could be used in F0 transform

    """

    def __init__(self, coef=1.2):
        self.coef = coef

    def convert(self, feature):
        return self.coef * feature
