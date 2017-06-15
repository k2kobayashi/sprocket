#! /usr/local/bin/python
# -*- coding: utf-8 -*-
#
# jnt.py
#   First ver.: 2017-06-15
#
#   Copyright 2017
#       Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
#   Distributed under terms of the MIT license.
#

"""


"""

import numpy as np
from multiprocessing import Pool

from dtw import dtw

from sprocket.util.yml import PairYML
from sprocket.util.hdf5 import HDF5
from sprocket.util.distance import melcd
from sprocket.util.delta import delta
from sprocket.util.extfrm import extfrm
from sprocket.model.GMM import GMMTrainer


"""
Memo: delta, extfrm, dtwまではパラレルで動く様にする．
jnt作成とGMMTrainはシングルの処理


ほしいもの: it0, it1, it2のjntとtwf, とGMM

"""


class JointFeatureExtractor(object):

    def __init__(self, yml, feature='mcep', mnum=1):

        # read pair-dependent yml file
        self.conf = PairYML(yml)

        if feature == 'mcep':
            self.distance = 'melcd'
        else:
            pass

        # open GMM for training and conversion
        self.gmm = GMMTrainer(yml)

    # TODO: these functions for open and close hdf5 will be moved to hdf.py
    def open_h5files(self):
        # read h5 files
        self.h5s = []
        self.num_files = len(self.conf.trfiles)
        for i in range(self.num_files):
            # open acoustic features
            orgh5 = HDF5(
                self.conf.h5dir + '/' + self.conf.trfiles[i][0] + '.h5', mode="r")
            tarh5 = HDF5(
                self.conf.h5dir + '/' + self.conf.trfiles[i][1] + '.h5', mode="r")
            self.h5s.append([orgh5, tarh5])
        return

    def close_h5files(self):
        # close hdf5 files
        for i in range(self.num_files):
            self.h5s[i][0].close()
            self.h5s[i][1].close()
        return

    def estimate(self):
        itnum = 0
        print(str(itnum) + '-th iteration start.')

        # open h5list files
        self.open_h5files()

        # create joint feature vector
        jnt = self._get_joint_feature_vector(itnum)

        # iterative GMM training
        while (itnum < self.conf.n_iter):
            itnum += 1
            print(str(itnum) + '-th iteration start.')
            # train GMM with joint feature vectors
            self.gmm.train(jnt)

            # dtw with converted feature if itnum > 1
            jnt = self._get_joint_feature_vector(itnum)

        # close hdf5 files
        self.close_h5files()

        return

    def _get_joint_feature_vector(self, itnum):
        for i in range(self.num_files):
            jdata = self._get_joint_feature_file(
                itnum, self.h5s[i][0], self.h5s[i][1])
            # create joint feature matrix
            if i == 0:
                jnt = jdata
            else:
                jnt = np.r_[jnt, jdata]
        return jnt

    def _get_joint_feature_file(self, itnum, orgh5, tarh5):
        # get delta and extract silence frame
        orgdata = calculate_extsddata(
            orgh5.read('mcep'), orgh5.read('npow'))
        tardata = calculate_extsddata(
            tarh5.read('mcep'), tarh5.read('npow'))

        if itnum == 0:
            # estimate twf function
            dist, _, _, twf = self._estimate_twf(orgdata, tardata)
        else:
            # TODO: convert acoustic feature of original
            # # conversion
            # conv = self.gmm.convert(orgdata)
            # # get delta and extract silence frame for converted
            # convdata = calculate_extsddata(conv, orgh5.read('npow'))
            # copy org data for debug (TODO: will be removed)
            convdata = orgdata

            # twf estimation between conv and tar
            dist, _, _, twf = self._estimate_twf(convdata, tardata)

        # TODO: save twf file (label: itnum, filename)
        print('distortion [dB]: ' + str(dist))
        jdata = self.generate_joint_feature_from_twf(orgdata, tardata, twf)

        return jdata

    def generate_joint_feature_from_twf(self, orgdata, tardata, twf):
        return np.c_[orgdata[twf[0]], tardata[twf[1]]]

    def _estimate_twf(self, orgdata, tardata):
        """
        return: dist, cost, _, path
        """

        if self.distance == 'melcd':
            distance_func = lambda x, y: melcd(x, y)
        else:
            raise('other distance metrics does not support.')

        return dtw(orgdata, tardata, dist=distance_func)


def gjfv_wrapper(args):
    return JointFeatureExtractor._get_joint_feature_vector(*args)


def calculate_extsddata(data, npow):
    return extract_silence_frame(calculate_delta(data), npow)


def calculate_delta(data):
    return np.c_[data, delta(data)]


def extract_silence_frame(data, npow):
    return extfrm(npow, data)


def main():
    pass


if __name__ == '__main__':
    main()
