import unittest

import os
import numpy as np
from scipy.io import wavfile

from sprocket.speech import WSOLA

dirpath = os.path.dirname(os.path.realpath(__file__))
saveflag = False


class WSOLATest(unittest.TestCase):

    def test_wsola(self):
        path = dirpath + '/data/test16000.wav'
        fs, x = wavfile.read(path)
        for speech_rate in (0.5, 0.75, 1.0, 1.5, 2.0):
            wsola = WSOLA(fs, speech_rate)
            wsolaed_x = wsola.duration_modification(x)
            assert int(len(x) / speech_rate) == len(wsolaed_x)

            if saveflag:
                fpath = path + str(speech_rate) + '.wav'
                wavfile.write(fpath, fs, wsolaed_x.astype(np.int16))
