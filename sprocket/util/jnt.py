# -*- coding: utf-8 -*-



import os
import numpy as np
from fastdtw import fastdtw

from sprocket.util.hdf5 import HDF5
from sprocket.util.distance import melcd, normalized_melcd
from sprocket.util.delta import delta
from sprocket.util.extfrm import extfrm
from sprocket.model.GMM import GMMTrainer, GMMConvertor


class JointFeatureExtractor(object):

    """Joint feature extractor class
    This class offers to extract time-aligned joint feature vector of original and
    target speakers' acoustic feature.

    Parameters
    ---------
    feature : str, optional
        The type of acoustic feature

    """

    def __init__(self, feature='mcep', jnt_iter=3, pairdir=None,
                 ):
        # distance setting
        if feature == 'mcep':
            self.distance = 'melcd'
            self.sd = 1  # start dimension mcep[T, 1:]
        else:
            raise('distance metrics does not support.')

        # open GMM for training and conversion
        self.n_jntiter = jnt_iter
        self.pairdir = pairdir

    def set_GMM_parameter(self, n_mix=32, n_iter=100, covtype='full', cvtype='mlpg'):
        self.trgmm = GMMTrainer(n_mix=n_mix, n_iter=n_iter, covtype=covtype)
        self.cvgmm = GMMConvertor(n_mix=n_mix, covtype=covtype, cvtype=cvtype)

    def estimate(self, orgfeatlist, tarfeatlist, orgnpowlist, tarnpowlist):
        """Estimate joint feature vector

        Parameters
        ---------
        orgfeatlist : list, shape(`num_files`)
            List of feature vectors for original speaker

        tarfeatlist : list, shape(`num_files`)
            List of feature vectors for target speaker

        orgnpowlist : list, shape(`num_files`)
            List of npow for target speaker

        tarnpowlist : list, shape(`num_files`)
            List of npow for target speaker

        """

        assert len(orgfeatlist) == len(tarfeatlist)
        assert len(orgnpowlist) == len(tarnpowlist)
        assert len(orgfeatlist) == len(orgnpowlist)

        self.orgfeat = orgfeatlist
        self.tarfeat = tarfeatlist
        self.orgnpow = orgnpowlist
        self.tarnpow = tarnpowlist

        self.num_files = len(self.orgfeat)

        itnum = 0
        print(str(itnum) + '-th joint feature extraction starts.')

        # create joint feature over utterances
        jnt = self._get_joint_feature_matrix(itnum)

        # iterative twf estimation
        while (itnum < self.n_jntiter):
            itnum += 1
            print(str(itnum) + '-th joint feature extraction start.')

            # train and save GMM with joint feature vectors
            self.trgmm.train(jnt)
            self._save_gmm(itnum)
            self._open_gmm(itnum)

            # dtw with converted original and target
            jnt = self._get_joint_feature_matrix(itnum)

        # save GMM and jnt file
        self._save_gmm(itnum)
        self._save_jnt(jnt, itnum)

        return

    def _get_joint_feature_matrix(self, itnum):
        for i in range(self.num_files):
            jdata = self._get_joint_feature(i, itnum)

            # concatenate joint feature data into joint feature matrix
            if i == 0:
                jnt = jdata
            else:
                jnt = np.r_[jnt, jdata]
        return jnt

    def _get_joint_feature(self, i, itnum):
        # get delta and extract silence frame
        orgdata = calculate_extsddata(
            self.orgfeat[i][:, self.sd:], self.orgnpow[i])
        tardata = calculate_extsddata(
            self.tarfeat[i][:, self.sd:], self.tarnpow[i])

        if itnum == 0:
            # estimate twf function
            twf = estimate_twf(orgdata, tardata, self.distance)

            norm_mcd = normalized_melcd(orgdata[twf[0]], tardata[twf[1]])
        else:
            # estimate twf with conversion
            # conversion
            conv = self.cvgmm.convert(
                calculate_delta(self.orgfeat[i][:, self.sd:]))

            # get delta and extract silence frame for converted
            cvdata = calculate_extsddata(conv, self.orgnpow[i])

            # twf estimation between conv and tar
            twf = estimate_twf(cvdata, tardata, self.distance)

            norm_mcd = normalized_melcd(cvdata[twf[0]], tardata[twf[1]])

        # print distortion
        print('distortion [dB] for ' + str(i) + '-th file: ' + str(norm_mcd))

        # save twf file
        self._save_twf(str(i) + '-th', twf, itnum)

        # generate joint feature vector of a phrase
        jdata = generate_joint_feature_from_twf(orgdata, tardata, twf)

        return jdata

    def _save_twf(self, flbl, twf, itnum):
        # save twf file as txt
        twfdir = self.pairdir + '/twf/it' + str(itnum)
        if not os.path.exists(twfdir):
            os.makedirs(twfdir)
        twfpath = twfdir + '/' + flbl + '.twf'
        natwf = np.array([twf[0], twf[1]])
        np.savetxt(twfpath, (natwf.T), fmt='%d')

        return

    def _save_jnt(self, jnt, itnum):
        # save jnt file as hdf5
        jntdir = self.pairdir + '/jnt'
        if not os.path.exists(jntdir):
            os.makedirs(jntdir)
        jntpath = jntdir + '/it' + str(itnum) + '.h5'
        h5 = HDF5(jntpath, mode='w')
        h5.save(jnt, ext='jnt')
        h5.close()

        return

    def _save_gmm(self, itnum):
        # save jnt file as pkl
        gmmdir = self.pairdir + '/GMM'
        if not os.path.exists(gmmdir):
            os.makedirs(gmmdir)
        gmmpath = gmmdir + '/it' + str(itnum) + '_GMM.pkl'
        self.trgmm.save(gmmpath)

        return

    def _open_gmm(self, itnum):
        # save jnt file as pkl
        gmmdir = self.pairdir + '/GMM'
        gmmpath = gmmdir + '/it' + str(itnum) + '_GMM.pkl'
        self.cvgmm.open(gmmpath)

        return


def estimate_twf(orgdata, tardata, distance='melcd'):
    """
    return: dist, cost, _, path
    """

    if distance == 'melcd':
        def distance_func(x, y): return melcd(x, y)
    else:
        raise('other distance metrics does not support.')

    _, path = fastdtw(orgdata, tardata, dist=distance_func)
    twf = np.array(path).T

    return twf


def generate_joint_feature_from_twf(orgdata, tardata, twf):
    return np.c_[orgdata[twf[0]], tardata[twf[1]]]


def calculate_extsddata(data, npow):
    return extract_silence_frame(calculate_delta(data), npow)


def calculate_delta(data):
    return np.c_[data, delta(data)]


def extract_silence_frame(data, npow):
    return extfrm(npow, data)
