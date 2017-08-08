#!/usr/bin/env bash

# speaker setting
readonly ORG=$1
readonly TAR=$2
readonly FS=$3

# directory setting
readonly CONF_DIR=./conf
readonly LIST_DIR=./list
readonly SRC_DIR=./src
readonly DATA_DIR=./data

function check_configure() {
    local _target_yml=$1
    local _defaul_yml=$2
    if [ -e $_target_yml ]; then
        echo "$_target_yml exists."
    else
        echo "generate $_target_yml"
        cp $_defaul_yml $_target_yml
    fi
    return 0
}

function check_list() {
    local _target_list=$1
    local _wav_dir=$2
    if [ -e $_target_list ] ; then
        echo "$_target_list exists."
    else
        echo "generate $_target_list"
        for _wav_file in `\ls $_wav_dir`;
        do
            echo `basename $_wav_dir`/${_wav_file%.*} >> $_target_list
        done
    fi
    return 0
}

echo "##############################################################"
echo "### 1. check configure files                               ###"
echo "##############################################################"
# check speaker-dependent configure file
check_configure $CONF_DIR/speaker/$ORG.yml $CONF_DIR/default/speaker_default_$FS.yml
check_configure $CONF_DIR/speaker/$TAR.yml $CONF_DIR/default/speaker_default_$FS.yml

# check pair-dependent configure file
check_configure $CONF_DIR/pair/$ORG-$TAR.yml $CONF_DIR/default/pair_default.yml

echo "##############################################################"
echo "### 2. check list files                                    ###"
echo "##############################################################"
# check list files for original speaker
check_list $LIST_DIR/${ORG}_train.list $DATA_DIR/wav/${ORG}
check_list $LIST_DIR/${ORG}_eval.list $DATA_DIR/wav/${ORG}

# check list files for target speaker
check_list $LIST_DIR/${TAR}_train.list $DATA_DIR/wav/${TAR}
check_list $LIST_DIR/${TAR}_eval.list $DATA_DIR/wav/${TAR}

echo "##############################################################"
echo "### 3. create figures to define F0 range                   ###"
echo "##############################################################"
# get F0 range in each speaker
for _speaker in $ORG $TAR; do
    listf=$LIST_DIR/${_speaker}_train.list
    python $SRC_DIR/initialize_speaker.py \
        $_speaker \
        $listf \
        $DATA_DIR/wav \
        $CONF_DIR/figure
done
echo "# Please modify minf0 and maxf0 in yml files based on the histgram #"
