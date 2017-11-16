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


def get_extsddata(data, npow, power_threshold=-20):
    """Get power extract static and delta feature vector

    Paramters
    ---------
    data : array, shape (`T`, `dim`)
        Acoustic feature vector
    npow : array, shape (`T`)
        Normalized power vector
    power_threshold : float, optional,
        Power threshold
        Default set to -20

    Returns
    -------
    extsddata : array, shape (`T_new` `dim * 2`)
        Silence remove static and delta feature vector

    """

    extsddata = extfrm(static_delta(data), npow,
                       power_threshold=power_threshold)
    return extsddata


def get_aligned_data(org_data, tar_data, twf):
    """Get aligned joint feature vector

    Paramters
    ---------
    org_data : array, shape (`T_org`, `dim_org`)
        Acoustic feature vector of original speaker
    tar_data : array, shape (`T_tar`, `dim_tar`)
        Acoustic feature vector of target speaker
    twf : array, shape (`2`)
        Time warping function

    Returns
    -------
    jdata : array, shape (`T_new` `dim_org + dim_tar`)
        Joint feature vector between source and target

    """

    jdata = np.c_[org_data[twf[0]], tar_data[twf[1]]]
    return jdata


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
    org_codeaps = read_feats(args.org_list_file, h5_dir, ext='codeap')
    org_npows = read_feats(args.org_list_file, h5_dir, ext='npow')
    tar_mceps = read_feats(args.tar_list_file, h5_dir, ext='mcep')
    tar_codeaps = read_feats(args.tar_list_file, h5_dir, ext='codeap')
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
        org_extsdmcep = get_extsddata(org_mceps[i][:, sd:],
                                      org_npows[i],
                                      power_threshold=oconf.power_threshold)
        tar_extsdmcep = get_extsddata(tar_mceps[i][:, sd:],
                                      tar_npows[i],
                                      power_threshold=tconf.power_threshold)
        twf = estimate_twf(org_extsdmcep, tar_extsdmcep, distance='melcd')
        jsdmcep = get_aligned_data(org_extsdmcep, tar_extsdmcep, twf)
        mcd = melcd(org_extsdmcep[twf[0]], tar_extsdmcep[twf[1]])

        print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
        if i == 0:
            jnt = jsdmcep
        else:
            jnt = np.r_[jnt, jsdmcep]
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
            org_extsdmcep = get_extsddata(org_mceps[i][:, sd:],
                                          org_npows[i],
                                          power_threshold=oconf.power_threshold)
            tar_extsdmcep = get_extsddata(tar_mceps[i][:, sd:],
                                          tar_npows[i],
                                          power_threshold=tconf.power_threshold)
            cv_extsdmcep = get_extsddata(cvmcep,
                                         org_npows[i],
                                         power_threshold=oconf.power_threshold)
            twf = estimate_twf(cv_extsdmcep, tar_extsdmcep, distance='melcd')
            jsdmcep = get_aligned_data(org_extsdmcep, tar_extsdmcep, twf)
            mcd = melcd(cv_extsdmcep[twf[0]], tar_extsdmcep[twf[1]])

            print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
            if i == 0:
                jnt = jsdmcep
            else:
                jnt = np.r_[jnt, jsdmcep]
            twfs.append(twf)

            if itnum == pconf.jnt_n_iter:
                # extract codeap joint feature vector
                org_extsdcodeap = get_extsddata(org_codeaps[i],
                                                org_npows[i],
                                                power_threshold=oconf.power_threshold)
                tar_extsdcodeap = get_extsddata(tar_codeaps[i],
                                                tar_npows[i],
                                                power_threshold=tconf.power_threshold)
                jcodeap = get_aligned_data(
                    org_extsdcodeap, tar_extsdcodeap, twf)
                if i == 0:
                    jnt_codeap = jcodeap
                else:
                    jnt_codeap = np.r_[jnt_codeap, jcodeap]

        itnum += 1

    # save joint feature vector of mcep
    jnt_dir = os.path.join(args.pair_dir, 'jnt')
    if not os.path.exists(jnt_dir):
        os.makedirs(jnt_dir)
    jntpath = os.path.join(jnt_dir, 'it' + str(itnum) + '_jnt.h5')
    jnth5 = HDF5(jntpath, mode='w')
    jnth5.save(jnt, ext='mcep')
    jnth5.save(jnt_codeap, ext='codeap')
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
