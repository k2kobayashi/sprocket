import unittest

import os
from scipy.io import wavfile

from sprocket.util.shifter import Shifter

dirpath = os.path.dirname(os.path.realpath(__file__))

class ShifterTest(unittest.TestCase):

    def test_shifter(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        for f0rate in (0.5, 0.75, 1.0, 1.5, 2.0):
            shifter = Shifter(fs, f0rate=f0rate)
            transformed_x = shifter.f0transform(x)
            assert len(x) == len(transformed_x)
