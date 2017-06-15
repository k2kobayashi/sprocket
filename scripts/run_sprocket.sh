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
pair_dir=$data_dir/pair/$org-$tar

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

if [ 0 -eq 1 ] ; then
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
    mkdir -p $pair_dir
    if [ ! -e $pair_dir/$org-$tar.yml ] ; then
        cp $conf_dir/default/pair_default.yml $pair_dir/$org-$tar.yml
        echo "list:" >> $pair_dir/$org-$tar.yml
        echo "    trlist: $pair_dir/$org-${tar}_tr.list" >> $pair_dir/$org-$tar.yml
        echo "    evlist: $pair_dir/$org-${tar}_ev.list" >> $pair_dir/$org-$tar.yml
    fi
    # Initilization of the speaker pair
    python $src_dir/init_pair.py \
        $org \
        $tar \
        $wav_dir \
        $pair_dir
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
        $pair_dir/$org-$tar.yml
fi

# Joint feature extraction
if [ 1 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Estimate time warping function using GMM               ###"
    echo "##############################################################"
    # estimate a time-aligned joint feature vector of source and target
    python $src_dir/estimate_twf.py \
        $org \
        $tar \
        $pair_dir/$org-$tar.yml
fi

# GMM train
if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Train conversion model                                 ###"
    echo "##############################################################"
    # estimate GMM parameter using the joint feature vector
    python $src_dir/train.py \
        $org \
        $tar \
        $pair_dir
fi

# Conversion based on GMM
if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Conversion based on the trained models                 ###"
    echo "##############################################################"
    # convertsion based on the trained GMM
    python $src_dir/convert.py \
        $org \
        $tar \
        $pair_dir \
        $wav_dir
fi
