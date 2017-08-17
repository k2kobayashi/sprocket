#!/usr/bin/env bash

# speaker setting
readonly ORG=$1
readonly TAR=$2
readonly FS=$3

. variable_and_function.sh

echo "##############################################################"
echo "### 1. create initial list files                           ###"
echo "##############################################################"
# check list files for original speaker
create_list $LIST_DIR/${ORG}_train.list $DATA_DIR/wav/${ORG} || exit $?
create_list $LIST_DIR/${ORG}_eval.list $DATA_DIR/wav/${ORG} || exit $?

# check list files for target speaker
create_list $LIST_DIR/${TAR}_train.list $DATA_DIR/wav/${TAR} || exit $?
create_list $LIST_DIR/${TAR}_eval.list $DATA_DIR/wav/${TAR} || exit $?
echo "# Please modify train and eval list files, if you want. #"

echo "##############################################################"
echo "### 2. create configure files                              ###"
echo "##############################################################"
# check speaker-dependent configure file
create_configure $CONF_DIR/speaker/$ORG.yml $CONF_DIR/default/speaker_default_$FS.yml || exit $?
create_configure $CONF_DIR/speaker/$TAR.yml $CONF_DIR/default/speaker_default_$FS.yml || exit $?

# check pair-dependent configure file
create_configure $CONF_DIR/pair/$ORG-$TAR.yml $CONF_DIR/default/pair_default.yml || exit $?

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
echo "# Please modify f0 range values in speaker-dependent YAML files based on the figure #"
