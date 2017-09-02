from __future__ import division, print_function, absolute_import

import os
import unittest

import numpy as np
from sprocket.util.hdf5 import HDF5, read_feats

dirpath = os.path.dirname(os.path.realpath(__file__))
listf = os.path.join(dirpath, '/data/test.h5')


class hdf5FunctionsTest(unittest.TestCase):

    def test_HDF5(self):
        data1d = np.random.rand(100)
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

        # tset read_feats function
        listpath = os.path.join(dirpath, 'data/test.list')
        with open(listpath, 'w') as fp:
            fp.write('data/test')
        list1d = read_feats(listpath, dirpath, ext='1d')
        assert np.allclose(list1d[0], data1d)

        # remove files
        os.remove(path)
        os.remove(listpath)
