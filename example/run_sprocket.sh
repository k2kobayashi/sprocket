#! /bin/sh
#
# run_sprocket.sh
# Copyright (C) 2017 Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
# Distributed under terms of the MIT license.
#

# speaker setting
org=SF1
tar=TF1

# flag settings
STEP1=1 # get figures of f0 range
STEP2=1 # extract acoustic feature
STEP3=1 # estimate acoustic feature statistics
STEP4=1 # estimate time warping function and joint feature vector
STEP5=1 # train GMM
STEP6=1 # convert based on the trained GMM

# directory setting
conf_dir=./conf
list_dir=./list
src_dir=./src
data_dir=./data
pair_dir=$data_dir/pair/$org-$tar
mkdir -p $pair_dir

# check speaker-dependent yml file
if [ ! -e $conf_dir/$org.yml ] ; then
    cp $conf_dir/default/speaker_default.yml $conf_dir/$org.yml
fi
if [ ! -e $conf_dir/$tar.yml ] ; then
    cp $conf_dir/default/speaker_default.yml $conf_dir/$tar.yml
fi
# check speaker pair yml file
if [ ! -e $pair_dir/$org-$tar.yml ] ; then
    cp $conf_dir/default/pair_default.yml $pair_dir/$org-$tar.yml
fi

# check list file
if [ ! -e $list_dir/${org}_train.list ] && [ ! -e $list_dir/${tar}_train.list ] ; then
    echo "Please prepare training list files for $org and $tar."
    exit
fi
if [ ! -e $list_dir/${org}_eval.list ] && [ ! -e $list_dir/${tar}_eval.list ] ; then
    echo "Please prepare evaluation list files for $org and $tar."
    exit
fi

if [ $STEP1 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 1. Initialization of original and target speakers      ###"
    echo "##############################################################"
    # Initialize speakers
    for speaker in $org $tar; do
        listf=$list_dir/${speaker}_train.list
        python $src_dir/initialize_speaker.py \
            $speaker \
            $listf \
            $data_dir/wav \
            $pair_dir
    done
    echo "# Please modify minf0 and maxf0 in yml files based on the histgram #"
    exit
fi

if [ $STEP2 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 2. Extract acoustic features                           ###"
    echo "##############################################################"
    # Extract acoustic features consisting of F0, spc, ap, mcep, npow
    for speaker in $org $tar; do
        for flag in train eval; do
            ymlf=$conf_dir/${speaker}.yml
            listf=$list_dir/${speaker}_${flag}.list
            python $src_dir/extract_features.py \
                $speaker \
                $ymlf \
                $listf \
                $data_dir/wav \
                $pair_dir
        done
    done
fi

if [ $STEP3 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 3. Estimate acoustic feature statistics                ###"
    echo "##############################################################"
    # Estimate speaker-dependent statistics for F0 and mcep
    for speaker in $org $tar; do
        listf=$list_dir/${speaker}_train.list
        python $src_dir/estimate_feature_statistics.py \
            $speaker \
            $listf \
            $pair_dir
    done
fi

# Joint feature extraction
if [ $STEP4 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 4. Estimate time warping function and jnt              ###"
    echo "##############################################################"
    # Estimate a time-aligned joint feature vector of source and target
    python $src_dir/estimate_twf_and_jnt.py \
        $pair_dir/$org-$tar.yml \
        $list_dir/${org}_train.list \
        $list_dir/${tar}_train.list \
        $pair_dir
fi

# GMM train
if [ $STEP5 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 5. Train GMM                                           ###"
    echo "##############################################################"
    # estimate GMM parameter using the joint feature vector
    python $src_dir/train_GMM.py \
        $pair_dir/$org-$tar.yml \
        $pair_dir
fi

# Conversion based on GMM
if [ $STEP6 -eq 1 ] ; then
    echo "##############################################################"
    echo "### 6. Conversion based on the trained models              ###"
    echo "##############################################################"
    eval_list_file=$list_dir/${org}_eval.list
    # convertsion based on the trained GMM
    python $src_dir/convert.py \
        $org \
        $tar \
        $conf_dir/$org.yml \
        $pair_dir/$org-$tar.yml \
        $eval_list_file \
        $data_dir/wav \
        $pair_dir
    python $src_dir/convert.py \
        -gmmmode diff \
        $org \
        $tar \
        $conf_dir/$org.yml \
        $pair_dir/$org-$tar.yml \
        $eval_list_file \
        $data_dir/wav \
        $pair_dir
fi
