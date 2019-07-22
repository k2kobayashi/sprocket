from __future__ import absolute_import, division, print_function

import os
import sys
import unittest
from pathlib import Path

import numpy as np

from sprocket.util.hdf5 import HDF5

dirpath = os.path.dirname(os.path.realpath(__file__))
listf = os.path.join(dirpath, '/data/test.h5')


class hdf5FunctionsTest(unittest.TestCase):

    def test_HDF5(self):
        data1d = np.random.rand(100)
        data1d2 = np.random.rand(50)
        data2d = np.random.rand(100).reshape(50, 2)

        # write test
        path = os.path.join(dirpath, 'data/test.h5')
        h5 = HDF5(path, 'w')
        h5.save(data1d, '1d')
        h5.save(data2d, '2d')
        h5.close()

        # open test
        tmph5 = HDF5(path, 'r')
        tmp1d = tmph5.read(ext='1d')
        tmp2d = tmph5.read(ext='2d')
        tmph5.close()
        assert np.allclose(tmp1d, data1d)
        assert np.allclose(tmp2d, data2d)

        # open test with 'with' statement
        with HDF5(path, 'r') as h5_with:
            assert np.allclose(h5_with.read('1d'), data1d)

        # read/write and replace test
        h5 = HDF5(path, 'a')
        tmp1d = h5.read(ext='1d')
        h5.save(data1d2, '1d')
        tmp1d2 = h5.read(ext='1d')
        h5.close()
        assert np.allclose(tmp1d, data1d)
        assert np.allclose(tmp1d2, data1d2)

        # remove files
        os.remove(path)

    def test_HDF5_current_dir(self):
        listf_current = os.path.split(listf)[-1]
        data1d = np.random.rand(50)
        try:
            h5_write = HDF5(Path(listf_current) if sys.version_info >= (3,6) else listf_current, 'w')
            h5_write.save(data1d, '1d')
            h5_write.close()
            h5_read = HDF5(os.curdir + os.sep + listf_current, 'r')
            read_data1d = h5_read.read(ext='1d')
            h5_read.close()
            assert np.allclose(data1d, read_data1d)
        except: # pragma: no cover
            raise
        finally:
            os.remove(listf_current)

