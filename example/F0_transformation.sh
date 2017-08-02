#! /bin/sh
#
# F0_transformation.sh
# Copyright (C) 2017 Kazuhiro KOBAYASHI <kobayashi.kazuhiro@g.sp.m.is.nagoya-u.ac.jp>
#
# Distributed under terms of the MIT license.
#

# speaker setting
org=SF1
tar=TM1

# flag settings
STEP1=0 # get figures of f0 range
STEP2=1 # F0 transformation of original waveform

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

# check list file
if [ ! -e $list_dir/${org}_train.list ] && [ ! -e $list_dir/${tar}_train.list ] ; then
    echo "Please prepare training list files for $org and $tar."
    exit
fi
if [ ! -e $list_dir/${org}_eval.list ] ; then
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
    echo "### 2. F0 transformation of original waveform              ###"
    echo "##############################################################"
    # transform F0 of waveform of original speaker
    python $src_dir/f0_transformation.py \
        $org \
        $conf_dir/${org}.yml \
        $conf_dir/${tar}.yml \
        $list_dir/${org}_train.list \
        $list_dir/${org}_eval.list \
        $list_dir/${tar}_train.list \
        $data_dir/wav
fi
