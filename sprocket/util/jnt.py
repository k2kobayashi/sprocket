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
        # self.gmm = GMMTrainer()

    def estimate(self):
        itnum = 0
        num_files = len(self.conf.trfiles)

        jntlist = []
        for i in range(num_files):
            print(self.conf.h5dir + '/' + self.conf.trfiles[i][0])
            # read acoustic features
            orgh5 = HDF5(
                self.conf.h5dir + '/' + self.conf.trfiles[i][0] + '.h5', mode="r")
            tarh5 = HDF5(
                self.conf.h5dir + '/' + self.conf.trfiles[i][1] + '.h5', mode="r")

            # extract joint feature vector
            jntlist.append(self._get_joint_feature_vector(itnum, orgh5, tarh5))

            orgh5.close()
            tarh5.close()

        print (jntlist)
        print (jntlist[i] for i in range(num_files))

        jnt = np.r_[(jntlist[i] for i in range(num_files))]


        # concatenate jnts
        if i == 0:
            jnt = jfeature
        else:
            jnt = np.c_[jnt, jfeature]



        # iterative GMM training
        while (itnum > self.conf.n_iter):
            itnum += 1
            # # train GMM with joint feature vectors
            # self.gmm.train(jnt)

            jfeature = self._get_joint_feature_vector(itnum, orgh5, tarh5)

            # concatenate jfeatures
            # jnt = np.c_[jfeature]
            jnt = jfeature

        # close HDF5 files

        return

    def _get_joint_feature_vector(self,
                                  itnum,
                                  orgh5,
                                  tarh5,
                                  convdata=None,
                                  ):
        if itnum == 0:
            # dtw with original and target feature vector
            # get delta and extract silence frame
            orgdata = calculate_extsddata(
                orgh5.read('mcep'), orgh5.read('npow'))
            tardata = calculate_extsddata(
                tarh5.read('mcep'), tarh5.read('npow'))

            # estimate twf function
            dist, _, _, path = self._estimate_twf(orgdata, tardata)
            print('Distance is' + dist)

            # create joint feature vector
            jdata = twf

        else:
            # dtw with converted original and target feature vector

            # # conversion
            # conv = self.gmm.convert(orgdata)

            # # get delta adn extract silence frame for converted
            # convdata = calculate_extsddata(conv, orgh5.read('npow'))

            # copy org data for debug
            convdata = orgdata

            # twf estimation between conv and tar
            jfeature = self._estimate_twf(convdata, tardata)

        return jfeature

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
