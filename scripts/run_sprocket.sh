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
nproc=7 # # of multi-proceccing cores

if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Copy default files for original and target speakr      ###"
    echo "##############################################################"
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
        python $src_dir/initialize_spkr.py \
            -m $nproc \
            $spkr \
            $conf_dir \
            $wav_dir
    done
fi

if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Extract features of original and target speakers       ###"
    echo "##############################################################"
    # Feature extraction
    for spkr in $org $tar; do
        python $src_dir/extract_features.py \
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
        echo "    pair: $pair_dir " >> $pair_dir/$org-$tar.yml
        echo "list:" >> $pair_dir/$org-$tar.yml
        echo "    trlist: $pair_dir/$org-${tar}_tr.list" >> $pair_dir/$org-$tar.yml
        echo "    evlist: $pair_dir/$org-${tar}_ev.list" >> $pair_dir/$org-$tar.yml
    fi
    # Initilization of the speaker pair
    python $src_dir/initialize_pair.py \
        $org \
        $tar \
        $wav_dir \
        $pair_dir
fi

if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Estimate acoustic feature statistics                   ###"
    echo "##############################################################"
    # calculate speaker-dependent statistics for F0 and mcep
    python $src_dir/estimate_feature_statistics.py \
        $org \
        $tar \
        $pair_dir/$org-$tar.yml
fi

# Joint feature extraction
if [ 0 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Estimate time warping function using GMM               ###"
    echo "##############################################################"
    # estimate a time-aligned joint feature vector of source and target
    python $src_dir/estimate_twf.py \
        -m $nproc \
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
    python $src_dir/train_GMM.py \
        $org \
        $tar \
        $pair_dir/$org-$tar.yml
fi

# Conversion based on GMM
if [ 1 -eq 1 ] ; then
    echo "##############################################################"
    echo "### Conversion based on the trained models                 ###"
    echo "##############################################################"
    # convertsion based on the trained GMM
    python $src_dir/convert.py \
        $org \
        $tar \
        $conf_dir/$tar.yml \
        $pair_dir/$org-$tar.yml
fi
