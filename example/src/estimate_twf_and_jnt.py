#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Estimate joint feature vector of the speaker pair using GMM

"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import numpy as np

from yml import PairYML

from sprocket.util import read_feats, sddata, extfrm, estimate_twf, melcd
from sprocket.model.GMM import GMMTrainer, GMMConvertor


def get_aligned_jointdata(orgdata, orgnpow, tardata, tarnpow, cvdata=None):
    # extract extsddata
    org_extsddata = extfrm(sddata(orgdata), orgnpow)
    tar_extsddata = extfrm(sddata(tardata), tarnpow)

    if cvdata is None:
        # calculate twf and mel-cd
        twf = estimate_twf(org_extsddata, tar_extsddata, distance='melcd')
        mcd = melcd(org_extsddata[twf[0]], tar_extsddata[twf[1]])
    else:
        # calculate twf and mel-cd with converted data
        cv_extsddata = extfrm(sddata(cvdata), orgnpow)
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
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('org_list_file', type=str,
                        help='List file of original speaker')
    parser.add_argument('tar_list_file', type=str,
                        help='List file of target speaker')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args(argv)

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
    sd = 1
    num_files = len(org_mceps)
    print(str(itnum) + '-th joint feature extraction starts.')

    # first iteration
    for i in range(num_files):
        jdata, twf, mcd = get_aligned_jointdata(org_mceps[i][:, sd:], org_npows[i],
                                                tar_mceps[i][:, sd:], tar_npows[i])
        print('distortion [dB] for ' + str(i) + '-th file: ' + str(mcd))
        if i == 0:
            jnt = jdata
        else:
            jnt = np.r_[jnt, jdata]
    itnum += 1

    # second through final iteration
    while itnum < pconf.jnt_n_iter + 1:
        print(str(itnum) + '-th joint feature extraction starts.')
        # train GMM
        trgmm = GMMTrainer(n_mix=pconf.GMM_mcep_n_mix,
                           n_iter=pconf.GMM_mcep_n_iter,
                           covtype=pconf.GMM_mcep_covtype)
        trgmm.train(jnt)

        cvgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                             covtype=pconf.GMM_mcep_covtype,
                             cvtype=pconf.GMM_mcep_cvtype)
        cvgmm.open_from_trainer(trgmm)
        twfs = []
        for i in range(num_files):
            cvmcep = cvgmm.convert(sddata(org_mceps[i][:, sd:]))
            jdata, twf, mcd = get_aligned_jointdata(org_mceps[i][:, sd:],
                                                    org_npows[i],
                                                    tar_mceps[i][:, sd:],
                                                    tar_npows[i],
                                                    cvdata=cvmcep)
            print('distortion [dB] for ' + str(i) + '-th file: ' + str(mcd))
            if i == 0:
                jnt = jdata
            else:
                jnt = np.r_[jnt, jdata]
            twfs.append(twf)

        itnum += 1

    # save files in final steps
    # save jnt
    # save GMM
    # save twf


if __name__ == '__main__':
    main()
