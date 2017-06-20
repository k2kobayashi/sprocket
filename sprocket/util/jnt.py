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
twf and joint feature extraction

"""

import os
import numpy as np
# from multiprocessing import Pool

from dtw import dtw

from sprocket.util.hdf5 import HDF5, open_h5files, close_h5files
from sprocket.util.distance import melcd
from sprocket.util.delta import delta
from sprocket.util.extfrm import extfrm
from sprocket.model.GMM import GMMTrainer


class JointFeatureExtractor(object):

    def __init__(self, conf, feature='mcep', mnum=1):

        # copy parameters
        self.conf = conf
        self.mnum = mnum

        # distance setting
        if feature == 'mcep':
            self.distance = 'melcd'
        else:
            raise('distance metrics does not support.')

        # open GMM for training and conversion
        self.gmm = GMMTrainer(conf)

    def estimate(self):
        itnum = 0
        print(str(itnum) + '-th joint feature extraction starts.')

        # open h5list files
        self.h5s = open_h5files(self.conf, mode='tr')
        self.num_files = len(self.h5s)

        # create joint feature over utterances
        jnt = self._get_joint_feature_matrix(itnum)

        # save jnt file (label: $pair_dir/jnt/itnum.mat)
        self._save_jnt(jnt, itnum)

        # iterative twf estimation
        while (itnum < self.conf.n_jntiter):
            itnum += 1
            print(str(itnum) + '-th joint feature extraction start.')

            # train and save GMM with joint feature vectors
            self.gmm.train(jnt)

            # dtw with converted original and target
            jnt = self._get_joint_feature_matrix(itnum)

            # save GMM and jnt file
            self._save_gmm(itnum)
            self._save_jnt(jnt, itnum)

        # close hdf5 files
        close_h5files(self.h5s)

        return

    def read_jnt(self):
        jntdir = self.conf.pairdir + '/jnt'
        jntpath = jntdir + '/it' + str(self.conf.n_jntiter - 1) + '.h5'

        if not os.path.exists(jntpath):
            raise('joint feature files does not exists.')

        h5 = HDF5(jntpath, mode='r')
        jnt = h5.read(ext='mat')
        h5.close()

        return jnt

    def generate_joint_feature_from_twf(self, orgdata, tardata, twf):
        return np.c_[orgdata[twf[0]], tardata[twf[1]]]

    # TODO: will be modified to multiprocessing
    def _get_joint_feature_matrix(self, itnum):
        for i in range(self.num_files):
            jdata = self._get_joint_feature_file(
                itnum, self.h5s[i][0], self.h5s[i][1])

            # concatenate joint feature data into joint feature matrix
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
            # conversion
            conv = self.gmm.convert(calculate_delta(orgh5.read('mcep')))

            # get delta and extract silence frame for converted
            convdata = calculate_extsddata(conv, orgh5.read('npow'))

            # twf estimation between conv and tar
            dist, _, _, twf = self._estimate_twf(convdata, tardata)

        # print distortion
        print('distortion [dB] for ' + orgh5.flbl + ': ' + str(dist))

        # save twf file
        self._save_twf(orgh5.flbl, twf, itnum)

        # generate joint feature vector of a phrase
        jdata = self.generate_joint_feature_from_twf(orgdata, tardata, twf)

        return jdata

    def _estimate_twf(self, orgdata, tardata):
        """
        return: dist, cost, _, path
        """

        if self.distance == 'melcd':
            distance_func = lambda x, y: melcd(x, y)
        else:
            raise('other distance metrics does not support.')

        return dtw(orgdata, tardata, dist=distance_func)

    def _save_twf(self, flbl, twf, itnum):
        twfdir = self.conf.pairdir + '/twf/it' + str(itnum)
        if not os.path.exists(twfdir):
            os.makedirs(twfdir)
        twfpath = twfdir + '/' + flbl + '.twf'
        natwf = np.array([twf[0], twf[1]])
        np.savetxt(twfpath, (natwf.T), fmt='%d')

        return

    def _save_jnt(self, jnt, itnum):
        jntdir = self.conf.pairdir + '/jnt'
        if not os.path.exists(jntdir):
            os.makedirs(jntdir)
        jntpath = jntdir + '/it' + str(itnum) + '.h5'
        h5 = HDF5(jntpath, mode='w')
        h5.save(jnt, ext='mat')
        h5.close()

        return

    def _save_gmm(self, itnum):
        gmmdir = self.conf.pairdir + '/GMM'
        if not os.path.exists(gmmdir):
            os.makedirs(gmmdir)
        gmmpath = gmmdir + '/it' + str(itnum) + '_GMM.pkl'
        self.gmm.save(gmmpath)

        return


def gjfv_wrapper(args):
    return JointFeatureExtractor._get_joint_feature_matrix(*args)


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
