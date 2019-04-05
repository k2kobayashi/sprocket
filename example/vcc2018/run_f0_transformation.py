#!/usr/bin/env python3

"""An example script to transform F0

Usage: run_f0_transformation.py [-h] [--f0rate <f0rate>] [--ev] SOURCE TARGET

Options:
    -h, --help   Show the help
    --f0rate <f0rate>  F0 transformation ratio, [default: -1]
    --ev           Transform waveform only for evaluation list
    SOURCE         The name of speaker
                   whose voice you would like to convert from
    TARGET         The name of speaker whose voice you would like to convert to

Note:
    At least one of the options that designates steps
    that are to be executed is required.
"""

from pathlib import Path
from docopt import docopt

from sprocket.bin import f0_transformation  # isort:skip # pylint: disable=C0413
from sprocket.util.misc import list_lengths_are_all_same


LIST_SUFFIXES = {
    use: "_" + use + ".list" for use in ("train", "eval")}

EXAMPLE_ROOT_DIR = Path(__file__).parent
CONF_DIR = EXAMPLE_ROOT_DIR / "conf"
DATA_DIR = EXAMPLE_ROOT_DIR / "data"
LIST_DIR = EXAMPLE_ROOT_DIR / "list"
WAV_DIR = DATA_DIR / "wav"

if __name__ == "__main__":
    args = docopt(__doc__)  # pylint: disable=invalid-name
    LABELS = {label: args[label.upper()] for label in ("source", "target")}
    SOURCE_TARGET_PAIR = LABELS["source"] + "-" + LABELS["target"]
    PAIR_DIR = DATA_DIR / "pair" / SOURCE_TARGET_PAIR
    LIST_FILES = {
        speaker_part: {
            use: LIST_DIR / (speaker_label + LIST_SUFFIXES[use])
            for use in ("train", "eval")}
        for speaker_part, speaker_label in LABELS.items()}
    SPEAKER_CONF_FILES = {
        part: CONF_DIR / "speaker" / (label + ".yml")
        for part, label in LABELS.items()}
    PAIR_CONF_FILE = CONF_DIR / "pair" / (SOURCE_TARGET_PAIR + ".yml")
    f0rate = args['--f0rate']
    evlist_only = args['--ev']

    # check list file
    for use in ("train", "eval"):
        list_lengths_are_all_same(
            LIST_FILES["source"][use], LIST_FILES["target"][use])

    print("### 1. F0 transformation of original waveform ###")
    # transform F0 of waveform of original speaker
    if evlist_only:
        f0_transformation.main("--f0rate", f0rate, "--evlist",
                               LABELS["source"], str(
                                   SPEAKER_CONF_FILES["source"]),
                               str(SPEAKER_CONF_FILES["target"]),
                               str(LIST_FILES["source"]["train"]),
                               str(LIST_FILES["source"]["eval"]),
                               str(LIST_FILES["target"]["train"]),
                               str(WAV_DIR))
    else:
        f0_transformation.main("--f0rate", f0rate,
                               LABELS["source"], str(
                                   SPEAKER_CONF_FILES["source"]),
                               str(SPEAKER_CONF_FILES["target"]),
                               str(LIST_FILES["source"]["train"]),
                               str(LIST_FILES["source"]["eval"]),
                               str(LIST_FILES["target"]["train"]),
                               str(WAV_DIR))

    print("# F0 transformed waveforms are generated #")
