import unittest

import os
import numpy as np
from scipy.io import wavfile

from sprocket.util.shifter import Shifter

dirpath = os.path.dirname(os.path.realpath(__file__))
saveflag = False


class ShifterTest(unittest.TestCase):

    def test_shifter(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        for f0rate in (0.5, 0.75, 1.0, 1.5, 2.0):
            if f0rate < 1:
                completion = True
            else:
                completion = False
            shifter = Shifter(fs, f0rate=f0rate, completion=completion)
            transformed_x = shifter.f0transform(x)
            assert len(x) == len(transformed_x)

        if saveflag:
            fpath = path + str(f0rate) + '.wav'
            wavfile.write(fpath, self.fs, transformed_x.astype(np.int16))
