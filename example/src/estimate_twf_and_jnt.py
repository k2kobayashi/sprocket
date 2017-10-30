#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate joint feature vector of the speaker pair using GMM

"""

import argparse
import os
import sys

import numpy as np
from sklearn.externals import joblib

from sprocket.model.GMM import GMMConvertor, GMMTrainer
from sprocket.util import HDF5, estimate_twf, extfrm, melcd, static_delta
from yml import SpeakerYML, PairYML

from .misc import read_feats


def get_aligned_jointdata(orgdata, orgnpow, tardata, tarnpow, cvdata=None,
                          orgpow_threshold=-20, tarpow_threshold=-20):
    """Get aligment between features

    Paramters
    ---------
    orgdata : array, shape (`T_org`, `dim`)
        Acoustic feature of source speaker
    orgnpow : array, shape (`T_org`)
        Normalized power of soruce speaker
    orgdata : array, shape (`T_tar`, `dim`)
        Acoustic feature of target speaker
    orgnpow : array, shape (`T_tar`)
        Normalized power of target speaker
    cvdata : array, optional, shape (`T_org`, `dim`)
        Converted acoustic feature from source into target
    orgpow_threshold : float, optional,
        Original speaker power threshold
        Default set to -20
    tarpow_threshold : float, optional,
        Target speaker power threshold
        Default set to -20

    Returns
    -------
    jdata : array, shape (`T_new` `dim * 2`)
        Joint feature vector between source and target
    twf : array, shape (`T_new`, `2`)
        Time warping function
    mcd : float,
        Mel-cepstrum distortion between source and target

    """

    # extract extsddata
    org_extsddata = extfrm(static_delta(orgdata), orgnpow,
                           power_threshold=orgpow_threshold)
    tar_extsddata = extfrm(static_delta(tardata), tarnpow,
                           power_threshold=tarpow_threshold)

    if cvdata is None:
        # calculate twf and mel-cd
        twf = estimate_twf(org_extsddata, tar_extsddata, distance='melcd')
        mcd = melcd(org_extsddata[twf[0]], tar_extsddata[twf[1]])
    else:
        if orgdata.shape != cvdata.shape:
            raise ValueError('Dimension mismatch between orgdata and cvdata: \
                             {} {}'.format(orgdata.shape, cvdata.shape))
        # calculate twf and mel-cd with converted data
        cv_extsddata = extfrm(static_delta(cvdata), orgnpow,
                              power_threshold=orgpow_threshold)
        twf = estimate_twf(cv_extsddata, tar_extsddata, distance='melcd')
        mcd = melcd(cv_extsddata[twf[0]], tar_extsddata[twf[1]])

    # concatenate joint feature data into joint feature matrix
    jdata = np.c_[org_extsddata[twf[0]], tar_extsddata[twf[1]]]

    return jdata, twf, mcd


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    description = 'estimate joint feature of source and target speakers'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org_yml', type=str,
                        help='Yml file of the original speaker')
    parser.add_argument('tar_yml', type=str,
                        help='Yml file of the target speaker')
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('org_list_file', type=str,
                        help='List file of original speaker')
    parser.add_argument('tar_list_file', type=str,
                        help='List file of target speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args(argv)

    # read speaker-dependent yml files
    oconf = SpeakerYML(args.org_yml)
    tconf = SpeakerYML(args.tar_yml)

    # read pair-dependent yml file
    pconf = PairYML(args.pair_yml)

    # read source and target features from HDF file
    h5_dir = os.path.join(args.pair_dir, 'h5')
    org_mceps = read_feats(args.org_list_file, h5_dir, ext='mcep')
    org_npows = read_feats(args.org_list_file, h5_dir, ext='npow')
    tar_mceps = read_feats(args.tar_list_file, h5_dir, ext='mcep')
    tar_npows = read_feats(args.tar_list_file, h5_dir, ext='npow')
    assert len(org_mceps) == len(tar_mceps)
    assert len(org_npows) == len(tar_npows)
    assert len(org_mceps) == len(org_npows)

    itnum = 1
    sd = 1  # start dimension for aligment of mcep
    num_files = len(org_mceps)
    print('{}-th joint feature extraction starts.'.format(itnum))

    # first iteration
    # dtw between original and target
    for i in range(num_files):
        jdata, _, mcd = get_aligned_jointdata(org_mceps[i][:, sd:],
                                              org_npows[i],
                                              tar_mceps[i][:, sd:],
                                              tar_npows[i],
                                              orgpow_threshold=oconf.power_threshold,
                                              tarpow_threshold=tconf.power_threshold
                                              )
        print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
        if i == 0:
            jnt = jdata
        else:
            jnt = np.r_[jnt, jdata]
    itnum += 1

    # second through final iteration
    # dtw between converted and target
    while itnum < pconf.jnt_n_iter + 1:
        print('{}-th joint feature extraction starts.'.format(itnum))
        # train GMM
        trgmm = GMMTrainer(n_mix=pconf.GMM_mcep_n_mix,
                           n_iter=pconf.GMM_mcep_n_iter,
                           covtype=pconf.GMM_mcep_covtype)
        trgmm.train(jnt)

        cvgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                             covtype=pconf.GMM_mcep_covtype)
        cvgmm.open_from_param(trgmm.param)
        twfs = []
        for i in range(num_files):
            cvmcep = cvgmm.convert(static_delta(org_mceps[i][:, sd:]),
                                   cvtype=pconf.GMM_mcep_cvtype)
            jdata, twf, mcd = get_aligned_jointdata(org_mceps[i][:, sd:],
                                                    org_npows[i],
                                                    tar_mceps[i][:, sd:],
                                                    tar_npows[i],
                                                    cvdata=cvmcep,
                                                    orgpow_threshold=oconf.power_threshold,
                                                    tarpow_threshold=tconf.power_threshold
                                                    )
            print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
            if i == 0:
                jnt = jdata
            else:
                jnt = np.r_[jnt, jdata]
            twfs.append(twf)

        itnum += 1

    # save joint feature vector
    jnt_dir = os.path.join(args.pair_dir, 'jnt')
    if not os.path.exists(jnt_dir):
        os.makedirs(jnt_dir)
    jntpath = os.path.join(jnt_dir, 'it' + str(itnum) + '_jnt.h5')
    jnth5 = HDF5(jntpath, mode='w')
    jnth5.save(jnt, ext='jnt')
    jnth5.close()

    # save GMM
    gmm_dir = os.path.join(args.pair_dir, 'GMM')
    if not os.path.exists(gmm_dir):
        os.makedirs(gmm_dir)
    gmmpath = os.path.join(gmm_dir, 'it' + str(itnum) + '_gmm.pkl')
    joblib.dump(trgmm.param, gmmpath)

    # save twf
    twf_dir = os.path.join(args.pair_dir, 'twf')
    if not os.path.exists(twf_dir):
        os.makedirs(twf_dir)
    with open(args.org_list_file, 'r') as fp:
        for line, twf in zip(fp, twfs):
            f = os.path.basename(line.rstrip())
            twfpath = os.path.join(
                twf_dir, 'it' + str(itnum) + '_' + f + '.h5')
            twfh5 = HDF5(twfpath, mode='w')
            twfh5.save(twf, ext='twf')
            twfh5.close()


if __name__ == '__main__':
    main()
