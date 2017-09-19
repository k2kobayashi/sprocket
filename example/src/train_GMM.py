#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train GMM and converted GV statistics

"""

from __future__ import absolute_import, division, print_function

import argparse
import os
import sys
import numpy as np
from sklearn.externals import joblib

from .misc import read_feats
from sprocket.model import GMMTrainer, GMMConvertor, GV
from sprocket.util import HDF5, static_delta
from yml import PairYML


def feature_conversion(pconf, org_mceps, gmm, gmmmode=None):
    """Conversion of mel-cesptrums

    Parameters
    ---------
    pconf : PairYAML,
        Class of PairYAML
    org_mceps : list, shape (`num_mceps`)
        List of mel-cepstrums
    gmm : sklean.mixture.GaussianMixture,
        Parameters of sklearn-based Gaussian mixture
    gmmmode : str, optional
        Flag to parameter generation technique
        `None` : Normal VC
        `diff` : Differential VC
        Default set to `None`

    Returns
    ---------
    cv_mceps : list , shape(`num_mceps`)
        List of converted mel-cespstrums

    """
    cvgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                         covtype=pconf.GMM_mcep_covtype,
                         gmmmode=gmmmode,
                         )
    cvgmm.open_from_param(gmm.param)

    sd = 1  # start dimension to convert
    cv_mceps = []
    for mcep in org_mceps:
        mcep_0th = mcep[:, 0]
        cvmcep = cvgmm.convert(static_delta(mcep[:, sd:]),
                               cvtype=pconf.GMM_mcep_cvtype)
        cvmcep = np.c_[mcep_0th, cvmcep]
        if gmmmode == 'diff':
            cvmcep[:, sd:] += mcep[:, sd:]
        elif gmmmode is not None:
            raise ValueError('gmmmode must be `None` or `diff`.')
        cv_mceps.append(cvmcep)

    return cv_mceps


def main(*argv):
    argv = argv if argv else sys.argv[1:]
    # Options for python
    description = 'Train GMM and converted GV statistics'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('org_list_file', type=str,
                        help='List file of original speaker')
    parser.add_argument('pair_yml', type=str,
                        help='Yml file of the speaker pair')
    parser.add_argument('pair_dir', type=str,
                        help='Directory path of h5 files')
    args = parser.parse_args(argv)

    # read pair-dependent yml file
    pconf = PairYML(args.pair_yml)

    # read joint feature vector
    jntf = os.path.join(args.pair_dir, 'jnt',
                        'it' + str(pconf.jnt_n_iter + 1) + '_jnt.h5')
    jnth5 = HDF5(jntf, mode='r')
    jnt = jnth5.read(ext='jnt')

    # train GMM for mcep using joint feature vector
    gmm = GMMTrainer(n_mix=pconf.GMM_mcep_n_mix, n_iter=pconf.GMM_mcep_n_iter,
                     covtype=pconf.GMM_mcep_covtype)
    gmm.train(jnt)

    # save GMM
    gmm_dir = os.path.join(args.pair_dir, 'model')
    if not os.path.exists(gmm_dir):
        os.makedirs(gmm_dir)
    gmmpath = os.path.join(gmm_dir, 'GMM.pkl')
    joblib.dump(gmm.param, gmmpath)
    print("Conversion model save into " + gmmpath)

    # calculate GV statistics of converted feature
    h5_dir = os.path.join(args.pair_dir, 'h5')
    org_mceps = read_feats(args.org_list_file, h5_dir, ext='mcep')

    cv_mceps = feature_conversion(pconf, org_mceps, gmm, gmmmode=None)
    diffcv_mceps = feature_conversion(pconf, org_mceps, gmm, gmmmode='diff')

    gv = GV()
    cvgvstats = gv.estimate(cv_mceps)
    diffcvgvstats = gv.estimate(diffcv_mceps)

    # open h5 files
    statspath = os.path.join(gmm_dir, 'cvgv.h5')
    cvgvh5 = HDF5(statspath, mode='w')
    cvgvh5.save(cvgvstats, ext='cvgv')
    cvgvh5.save(diffcvgvstats, ext='diffcvgv')
    print("Converted gvstats save into " + statspath)


if __name__ == '__main__':
    main()
