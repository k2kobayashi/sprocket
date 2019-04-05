#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Create HDF5 file to create joint feature vector

"""

import argparse
import os
import sys

from sprocket.util import HDF5
from sprocket.util.yml import PairYML
from sprocket.util.misc import read_feats


def read_twfs(listf, twf_dir, pconf):
    """read twf files in twf_dir

    Parameters
    ----------
    listf : str
        Path of list file
    twf_dir : str
        Directory path of twf
    pconf : PairYML
        Configures depending on speaker pair

    Returns
    -------
    twfs : list
        List of twfs
    """
    with open(listf, 'r') as fp:
        twfs = []
        for line in fp:
            f = os.path.basename(line.rstrip())
            twfpath = os.path.join(twf_dir, 'it' + str(pconf.jnt_n_iter)
                                   + '_' + f + '.h5')
            twfh5 = HDF5(twfpath, mode='a')
            twfs.append(twfh5.read(ext='twf'))
            twfh5.close()
    return twfs


def save_jfeats(jfeats, jfeat_dir, listf):
    """
    """
    for key, feats in jfeats.items():
        with open(listf, 'r') as fp:
            for line, feat in zip(fp, feats):
                f = os.path.basename(line.rstrip())
                jfeatpath = os.path.join(jfeat_dir, f + '.h5')
                jfeath5 = HDF5(jfeatpath, mode='a')
                jfeath5.save(feat, ext=key)
                jfeath5.close()


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
    pconf = PairYML(args.pair_yml)

    # read source and target features from HDF file
    h5_dir = os.path.join(args.pair_dir, 'h5')
    twf_dir = os.path.join(args.pair_dir, 'twf')
    jfeats = {}
    jfeats['org_f0'] = read_feats(args.org_list_file, h5_dir, ext='f0')
    jfeats['org_mcep'] = read_feats(args.org_list_file, h5_dir, ext='mcep')
    jfeats['org_npow'] = read_feats(args.org_list_file, h5_dir, ext='npow')
    jfeats['org_codeap'] = read_feats(args.org_list_file, h5_dir, ext='codeap')
    jfeats['tar_f0'] = read_feats(args.tar_list_file, h5_dir, ext='f0')
    jfeats['tar_mcep'] = read_feats(args.tar_list_file, h5_dir, ext='mcep')
    jfeats['tar_npow'] = read_feats(args.tar_list_file, h5_dir, ext='npow')
    jfeats['tar_codeap'] = read_feats(args.tar_list_file, h5_dir, ext='codeap')
    jfeats['twf'] = read_twfs(args.org_list_file, twf_dir, pconf)

    # save jfeat
    jfeat_dir = os.path.join(args.pair_dir, 'jfeat')
    if not os.path.exists(jfeat_dir):
        os.mkdir(jfeat_dir)
    save_jfeats(jfeats, jfeat_dir, args.org_list_file)


if __name__ == '__main__':
    main()
