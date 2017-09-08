#!/usr/bin/env python

"""An example script to transform F0

Usage: f0_transformation.py [-h] SOURCE TARGET

Options:
    -h, --help   Show the help
    SOURCE         The name of speaker
                   whose voice you would like to convert from
    TARGET         The name of speaker whose voice you would like to convert to

Note:
    At least one of the options that designates steps
    that are to be executed is required.
"""

from __future__ import division  # , unicode_literals
from __future__ import absolute_import, print_function

import os
import sys

from docopt import docopt
import six

from run_sprocket import list_lengths_are_all_same

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # isort:skip
from src.f0_transformation import main  # isort:skip # pylint: disable=C0413

if six.PY2:
    # pylint: disable=unused-import, redefined-builtin, import-error, line-too-long
    from future_builtins import ascii, filter, hex, map, oct, zip
    # pylint: disable=unused-import, redefined-builtin, import-error, line-too-long, ungrouped-imports
    from six.moves import range, input
if six.PY3:
    # pylint: disable=unused-import, redefined-builtin, import-error, line-too-long
    from six.moves import reduce

LIST_SUFFIXES = {
    use: "_" + use + ".list" for use in ("train", "eval")}

EXAMPLE_ROOT_DIR = os.path.dirname(__file__)
CONF_DIR = os.path.join(EXAMPLE_ROOT_DIR, "conf")
DATA_DIR = os.path.join(EXAMPLE_ROOT_DIR, "data")
LIST_DIR = os.path.join(EXAMPLE_ROOT_DIR, "list")
WAV_DIR = os.path.join(DATA_DIR, "wav")

if __name__ == "__main__":
    args = docopt(__doc__)  # pylint: disable=invalid-name
    LABELS = {label: args[label.upper()] for label in ("source", "target")}
    SOURCE_TARGET_PAIR = LABELS["source"] + "-" + LABELS["target"]
    PAIR_DIR = os.path.join(DATA_DIR, "pair",
                            SOURCE_TARGET_PAIR)
    LIST_FILES = {
        speaker_part: {
            use: os.path.join(LIST_DIR, speaker_label + LIST_SUFFIXES[use])
            for use in ("train", "eval")}
        for speaker_part, speaker_label in LABELS.items()}
    SPEAKER_CONF_FILES = {
        part: os.path.join(
            CONF_DIR, "speaker", label + ".yml")
        for part, label in LABELS.items()}
    PAIR_CONF_FILE = os.path.join(
        CONF_DIR, "pair", SOURCE_TARGET_PAIR + ".yml")

    # check list file
    for use in ("train", "eval"):
        list_lengths_are_all_same(
            LIST_FILES["source"][use], LIST_FILES["target"][use])

    print("### 1. F0 transformation of original waveform ###")
    # transform F0 of waveform of original speaker
    # python $SRC_DIR / f0_transformation.py \
    # $ORG \
    # $CONF_DIR / speaker /${ORG}.yml \
    # $CONF_DIR / speaker /${TAR}.yml \
    # $LIST_DIR /${ORG}_train.list \
    # $LIST_DIR /${ORG}_eval.list \
    # $LIST_DIR /${TAR}_train.list \
    # $DATA_DIR / wav
    main(LABELS["source"], os.path.join(SPEAKER_CONF_FILES["source"]),
         os.path.join(SPEAKER_CONF_FILES["target"]),
         LIST_FILES["source"]["train"],
         LIST_FILES["source"]["eval"], LIST_FILES["target"]["train"], WAV_DIR)
    print("# F0 transformed waveforms are generated #")
