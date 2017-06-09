#! /bin/sh
#
# run_sprocket.sh
# Copyright (C) 2017 Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
# Distributed under terms of the MIT license.
#

# arguments setting
org=clb
tar=slt

# directory setting
src_dir=./src
data_dir=./data
conf_dir=./configure
wav_dir=$data_dir/speaker/wav

# parameter setting
nproc=7

if [ 0 -eq 1 ] ; then
    echo "### Copy default files for original and target speakr ###"
    for spkr in $org $tar; do
        if [ ! -e $conf_dir/$spkr.yml ]; then
            cp $conf_dir/default/speaker_default.yml $conf_dir/$spkr.yml
        fi
    done
fi

if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Initialization of original and target speakers         ###"
    echo "##############################################################"
    # Initialization for speakers
    for spkr in $org $tar; do
        python $src_dir/init_spkr.py \
            -m $nproc \
            $spkr \
            $wav_dir \
            $conf_dir
    done
fi

if [ 1 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Feature extcation for original and target speakers     ###"
    echo "##############################################################"
    # Feature extraction
    for spkr in $org $tar; do
        python $src_dir/feature_extraction.py \
            -m $nproc \
            $spkr \
            $conf_dir/$spkr.yml \
            $wav_dir
    done
fi

if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Initialization of the speaker pair                     ###"
    echo "##############################################################"
    mkdir $data_dir/pair/$org-$tar
    cp $conf_dir/default/pair_default.yml $data_dir/pair/$org-$tar.yml
    # Initilization of the speaker pair
    python $src_dir/init_pair.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Estimate acoustic feature statistics                   ###"
    echo "##############################################################"
    # calculate speaker-dependent statistics such as F0, GV and MS
    python $src_dir/estimate_feature_stats.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# Joint feature extraction
if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Estimate joint feature vector using GMM                ###"
    echo "##############################################################"
    # estimate a time-aligned joint feature vector of source and target
    python $src_dir/estimate_jnt.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# GMM train
if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Train conversion model based on GMM                    ###"
    echo "##############################################################"
    # estimate GMM parameter using the joint feature vector
    python $src_dir/train_GMM.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi

# Conversion based on GMM
if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Conversion based on the trained GMM                    ###"
    echo "##############################################################"
    # convertsion based on the trained GMM
    python $src_dir/gmmmap.py \
        $org \
        $tar \
        $wav_dir \
        $conf_dir
fi
