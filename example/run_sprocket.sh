#! /bin/sh
#
# run_sprocket.sh
# Copyright (C) 2017 Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
# Distributed under terms of the MIT license.
#

# speaker setting
# org=SF1
# tar=TF1
org=$1
tar=$2

# flag settings
STEP1=0 # initialize speaker
STEP2=1 # feature extraction
STEP3=0 # feature statistics extraction
STEP4=0 # estimate twf and joint feature vector
STEP5=0 # GMM training
STEP6=0 # conversion

# directory setting
conf_dir=./conf
list_dir=./list
src_dir=./src
data_dir=./data
pair_dir=./data/pair/$org-$tar
mkdir -p $pair_dir

# check yml file
if [ ! -e $conf_dir/$org.yml ] ; then
    cp $conf_dir/default/speaker_default.yml $conf_dir/$org.yml
fi
if [ ! -e $conf_dir/$tar.yml ] ; then
    cp $conf_dir/default/speaker_default.yml $conf_dir/$tar.yml
fi

# check list file
if [ ! -e $list_dir/${org}_tr.list ] && [ ! -e $list_dir/${tar}_tr.list ] ; then
    echo "Please prepare training list files for $org and $tar."
fi
if [ ! -e $list_dir/${org}_ev.list ] ; then
    echo "Please prepare evaluation list files for $org."
fi

if [ $STEP1 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 1. Initialization of original and target speakers      ###"
    echo "##############################################################"
    # Initialize speakers
    for speaker in $org $tar; do
        listf=$list_dir/${speaker}_tr.list
        histgramf=$data_dir/f0histgram/${speaker}_f0range.png
        python $src_dir/initialize_speaker.py \
            $speaker \
            $listf \
            $data_dir/wav \
            $histgramf
    done
    echo "# Please modify minf0 and maxf0 in yml files based on the histgram #"
    exit
fi

if [ $STEP2 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 2. Extract features of original and target speakers    ###"
    echo "##############################################################"
    # Extract acoustic features including F0, spc, ap, mcep, npow
    for speaker in $org $tar; do
        ymlf=$conf_dir/${speaker}.yml
        for flag in tr ev; do
            listf=$list_dir/${speaker}_$flag.list
            python $src_dir/extract_features.py \
                $speaker \
                $ymlf \
                $listf \
                $data_dir/wav \
                $data_dir/h5
        done
    done
fi

if [ $STEP3 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 3. Estimate acoustic feature statistics                ###"
    echo "##############################################################"
    # Estimate speaker-dependent statistics for F0 and mcep
    for speaker in $org $tar; do
        listf=$list_dir/${speaker}_tr.list
        python $src_dir/estimate_feature_statistics.py \
            $speaker \
            $listf \
            $data_dir/h5 \
            $pair_dir
    done
fi

# copy pair default yml file if not exit.
if [ ! -e $pair_dir/$org-$tar.yml ] ; then
    cp $conf_dir/default/pair_default.yml $pair_dir/$org-$tar.yml
fi

# Joint feature extraction
if [ $STEP4 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 4. Estimate time warping function using GMM            ###"
    echo "##############################################################"
    # Estimate a time-aligned joint feature vector of source and target
    python $src_dir/estimate_twf.py \
        $list_dir/${org}_tr.list \
        $list_dir/${tar}_tr.list \
        $pair_dir \
        $data_dir/h5
fi

# GMM train
if [ $STEP5 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 5. Train conversion model                              ###"
    echo "##############################################################"
    # estimate GMM parameter using the joint feature vector
    python $src_dir/train_GMM.py \
        $pair_dir
fi

# Conversion based on GMM
if [ $STEP6 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 6. Conversion based on the trained models              ###"
    echo "##############################################################"
    # convertsion based on the trained GMM
    python $src_dir/convert.py \
        $org \
        $tar \
        $list_dir/${org}_ev.list \
        $data_dir/wav \
        $data_dir/h5 \
        $pair_dir
    python $src_dir/convert.py \
        -cvtype diff \
        $org \
        $tar \
        $list_dir/${org}_ev.list \
        $data_dir/wav \
        $data_dir/h5 \
        $pair_dir
fi
