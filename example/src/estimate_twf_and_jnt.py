#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate joint feature vector of the speaker pair using GMM

"""

import argparse
import os
import sys

from sprocket.model.GMM import GMMConvertor, GMMTrainer
from sprocket.util import HDF5, estimate_twf, melcd
from sprocket.util import static_delta, align_data

from yml import SpeakerYML, PairYML
from misc import read_feats, extsddata, transform_jnt


def get_alignment(odata, onpow, tdata, tnpow, opow=-20, tpow=-20,
                  sd=0, cvdata=None, given_twf=None, otflag=None,
                  distance='melcd'):
    """Get alignment between original and target

    Paramters
    ---------
    odata : array, shape (`T`, `dim`)
        Acoustic feature vector of original
    onpow : array, shape (`T`)
        Normalized power vector of original
    tdata : array, shape (`T`, `dim`)
        Acoustic feature vector of target
    tnpow : array, shape (`T`)
        Normalized power vector of target
    opow : float, optional,
        Power threshold of original
        Default set to -20
    tpow : float, optional,
        Power threshold of target
        Default set to -20
    sd : int , optional,
        Start dimension to be used for alignment
        Default set to 0
    cvdata : array, shape (`T`, `dim`), optional,
        Converted original data
        Default set to None
    given_twf : array, shape (`T_new`, `dim * 2`), optional,
        Alignment given twf
        Default set to None
    otflag : str, optional
        Alignment into the length of specification
        'org' : alignment into original length
        'tar' : alignment into target length
        Default set to None
    distance : str,
        Distance function to be used
        Default set to 'melcd'

    Returns
    -------
    jdata : array, shape (`T_new` `dim * 2`)
        Joint static and delta feature vector
    twf : array, shape (`T_new` `dim * 2`)
        Time warping function
    mcd : float,
        Mel-cepstrum distortion between arrays

    """

    oexdata = extsddata(odata[:, sd:], onpow,
                        power_threshold=opow)
    texdata = extsddata(tdata[:, sd:], tnpow,
                        power_threshold=tpow)

    if cvdata is None:
        align_odata = oexdata
    else:
        cvexdata = extsddata(cvdata, onpow,
                             power_threshold=opow)
        align_odata = cvexdata

    if given_twf is None:
        twf = estimate_twf(align_odata, texdata,
                           distance=distance, otflag=otflag)
    else:
        twf = given_twf

    jdata = align_data(oexdata, texdata, twf)
    mcd = melcd(align_odata[twf[0]], texdata[twf[1]])

    return jdata, twf, mcd


def align_feature_vectors(odata, onpows, tdata, tnpows, pconf,
                          opow=-100, tpow=-100, itnum=3, sd=0,
                          given_twfs=None, otflag=None):
    """Get alignment to create joint feature vector

    Paramters
    ---------
    odata : list, (`num_files`)
        List of original feature vectors
    onpows : list , (`num_files`)
        List of original npows
    tdata : list, (`num_files`)
        List of target feature vectors
    tnpows : list , (`num_files`)
        List of target npows
    opow : float, optional,
        Power threshold of original
        Default set to -100
    tpow : float, optional,
        Power threshold of target
        Default set to -100
    itnum : int , optional,
        The number of iteration
        Default set to 3
    sd : int , optional,
        Start dimension of feature vector to be used for alignment
        Default set to 0
    given_twf : array, shape (`T_new` `dim * 2`)
        Use given alignment while 1st iteration
        Default set to None
    otflag : str, optional
        Alignment into the length of specification
        'org' : alignment into original length
        'tar' : alignment into target length
        Default set to None

    Returns
    -------
    jdata : array, shape (`T_new` `dim * 2`)
        Joint static and delta feature vector
    twf : array, shape (`T_new` `dim * 2`)
        Time warping function
    mcd : float ,
        Mel-cepstrum distortion between arrays

    """
    it = 1
    num_files = len(odata)
    cvgmm, cvdata = None, None
    for it in range(1, itnum + 1):
        print('{}-th joint feature extraction starts.'.format(it))
        # alignment
        twfs, jfvs = [], []
        for i in range(num_files):
            if it == 1 and given_twfs is not None:
                gtwf = given_twfs[i]
            else:
                gtwf = None
            if it > 1:
                cvdata = cvgmm.convert(static_delta(odata[i][:, sd:]),
                                       cvtype=pconf.GMM_mcep_cvtype)
            jdata, twf, mcd = get_alignment(odata[i],
                                            onpows[i],
                                            tdata[i],
                                            tnpows[i],
                                            opow=opow,
                                            tpow=tpow,
                                            sd=sd,
                                            cvdata=cvdata,
                                            given_twf=gtwf,
                                            otflag=otflag)
            twfs.append(twf)
            jfvs.append(jdata)
            print('distortion [dB] for {}-th file: {}'.format(i + 1, mcd))
        jnt_data = transform_jnt(jfvs)

        if it != itnum:
            # train GMM, if not final iteration
            datagmm = GMMTrainer(n_mix=pconf.GMM_mcep_n_mix,
                                 n_iter=pconf.GMM_mcep_n_iter,
                                 covtype=pconf.GMM_mcep_covtype)
            datagmm.train(jnt_data)
            cvgmm = GMMConvertor(n_mix=pconf.GMM_mcep_n_mix,
                                 covtype=pconf.GMM_mcep_covtype)
            cvgmm.open_from_param(datagmm.param)
        it += 1
    return jfvs, twfs


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

    # dtw between original and target w/o 0th and silence
    print('## Alignment mcep w/o 0-th and silence ##')
    jmceps, twfs = align_feature_vectors(org_mceps,
                                         org_npows,
                                         tar_mceps,
                                         tar_npows,
                                         pconf,
                                         opow=oconf.power_threshold,
                                         tpow=tconf.power_threshold,
                                         itnum=pconf.jnt_n_iter,
                                         sd=1,
                                         )
    jnt_mcep = transform_jnt(jmceps)

    # create joint feature for codeap using given twfs
    print('## Alignment codeap into target length using given twf ##')
    org_codeaps = read_feats(args.org_list_file, h5_dir, ext='codeap')
    tar_codeaps = read_feats(args.tar_list_file, h5_dir, ext='codeap')
    jcodeaps = []
    for i in range(len(org_codeaps)):
        # extract codeap joint feature vector
        jcodeap, _, _ = get_alignment(org_codeaps[i],
                                      org_npows[i],
                                      tar_codeaps[i],
                                      tar_npows[i],
                                      opow=oconf.power_threshold,
                                      tpow=tconf.power_threshold,
                                      given_twf=twfs[i])
        jcodeaps.append(jcodeap)
    jnt_codeap = transform_jnt(jcodeaps)

    # save joint feature vectors
    jnt_dir = os.path.join(args.pair_dir, 'jnt')
    os.makedirs(jnt_dir, exist_ok=True)
    jntpath = os.path.join(jnt_dir, 'it' + str(pconf.jnt_n_iter) + '_jnt.h5')
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
                twf_dir, 'it' + str(pconf.jnt_n_iter) + '_' + f + '.h5')
            twfh5 = HDF5(twfpath, mode='w')
            twfh5.save(twf, ext='twf')
            twfh5.close()


if __name__ == '__main__':
    main()
