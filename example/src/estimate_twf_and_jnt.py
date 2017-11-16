#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate joint feature vector of the speaker pair using GMM

"""

import argparse
import os
import sys
import numpy as np

from sprocket.model.GMM import GMMConvertor, GMMTrainer
from sprocket.util import HDF5, estimate_twf, extfrm, melcd
from sprocket.util import static_delta, align_data

from yml import SpeakerYML, PairYML
from misc import read_feats


def extsddata(data, npow, power_threshold=-20):
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
    jnt_mcep = np.empty([0, (oconf.mcep_dim + tconf.mcep_dim) * 2])
    for i in range(num_files):
        org_exmcep = extsddata(org_mceps[i][:, sd:],
                               org_npows[i],
                               power_threshold=oconf.power_threshold)
        tar_exmcep = extsddata(tar_mceps[i][:, sd:],
                               tar_npows[i],
                               power_threshold=tconf.power_threshold)
        twf = estimate_twf(org_exmcep, tar_exmcep, distance='melcd')
        jmcep = align_data(org_exmcep, tar_exmcep, twf)
        mcd = melcd(org_exmcep[twf[0]], tar_exmcep[twf[1]])

        jnt_mcep = np.r_[jnt_mcep, jmcep]
        print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
    itnum += 1

    # second through final iteration
    # dtw between converted and target
    while itnum < pconf.jnt_n_iter + 1:
        print('{}-th joint feature extraction starts.'.format(itnum))
        # train GMM
        mcepgmm = GMMTrainer(n_mix=pconf.GMM_mcep_n_mix,
                             n_iter=pconf.GMM_mcep_n_iter,
                             covtype=pconf.GMM_mcep_covtype)
        mcepgmm.train(jnt_mcep)
        cvgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                             covtype=pconf.GMM_mcep_covtype)
        cvgmm.open_from_param(mcepgmm.param)

        twfs = []
        jnt_mcep = np.empty([0, (oconf.mcep_dim + tconf.mcep_dim) * 2])
        for i in range(num_files):
            cvmcep = cvgmm.convert(static_delta(org_mceps[i][:, sd:]),
                                   cvtype=pconf.GMM_mcep_cvtype)
            org_exmcep = extsddata(org_mceps[i][:, sd:],
                                   org_npows[i],
                                   power_threshold=oconf.power_threshold)
            tar_exmcep = extsddata(tar_mceps[i][:, sd:],
                                   tar_npows[i],
                                   power_threshold=tconf.power_threshold)
            cv_extmcep = extsddata(cvmcep,
                                   org_npows[i],
                                   power_threshold=oconf.power_threshold)
            twf = estimate_twf(cv_extmcep, tar_exmcep, distance='melcd')
            jmcep = align_data(org_exmcep, tar_exmcep, twf)
            mcd = melcd(cv_extmcep[twf[0]], tar_exmcep[twf[1]])

            twfs.append(twf)
            jnt_mcep = np.r_[jnt_mcep, jmcep]
            print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))

        itnum += 1

    # create joint feature for codeap
    org_codeaps = read_feats(args.org_list_file, h5_dir, ext='codeap')
    tar_codeaps = read_feats(args.tar_list_file, h5_dir, ext='codeap')
    for i in range(num_files):
        # extract codeap joint feature vector in final iteration
        org_extcodeap = extsddata(org_codeaps[i],
                                  org_npows[i],
                                  power_threshold=oconf.power_threshold)
        tar_extcodeap = extsddata(tar_codeaps[i],
                                  tar_npows[i],
                                  power_threshold=tconf.power_threshold)
        jcodeap = align_data(org_extcodeap, tar_extcodeap, twfs[i])
        if i == 0:
            jnt_codeap = jcodeap
        else:
            jnt_codeap = np.r_[jnt_codeap, jcodeap]

    # save joint feature vectors
    jnt_dir = os.path.join(args.pair_dir, 'jnt')
    os.makedirs(jnt_dir, exist_ok=True)
    jntpath = os.path.join(jnt_dir, 'it' + str(itnum) + '_jnt.h5')
    jnth5 = HDF5(jntpath, mode='w')
    jnth5.save(jnt_mcep, ext='mcep')
    jnth5.save(jnt_codeap, ext='codeap')
    jnth5.close()

    # save twfs
    twf_dir = os.path.join(args.pair_dir, 'twf')
    os.makedirs(twf_dir, exist_ok=True)
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
