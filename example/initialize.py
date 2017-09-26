#!/usr/bin/env python3

"""An example script to initialize audio lists and speaker configurations.

Usage: initialize.py [-h] [-1] [-2] [-3] SOURCE TARGET SAMPLING_RATE

Options:
    -h, --help     Show the help
    -1, --step1    Execute step1 (Generation of initial list files)
    -2, --step2    Execute step2 (Generation of configure files)
    -3, --step3    Execute step3 (Estimation of F0 ranges)
    SOURCE         The name of speaker
                   whose voice you would like to convert from
    TARGET         The name of speaker whose voice you would like to convert to
    SAMPLING_RATE  The sampling rate of WAV files of voices
"""

import os
import shutil
import sys
from pathlib import Path

from docopt import docopt

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))  # isort:skip
from src import initialize_speaker  # isort:skip # pylint: disable=C0413


def create_configure(dest, base, exist_ok=True):
    """Creates a configuration file based on a template file.

    This does not overwrite the existing one.

    Parameters
    ----------
    dest : str or path-like
        The path of the configuration file you are creating.
    base : str or path-like
        The path of the template configure file.
    exist_ok : bool
        If `False`, this function throws
        `IOError` (Python 2.7) or `FileExistsError` (Python 3 or later)
        when `dest` is already created.

    Raises
    ------
    FileExistsError
        If `exist_ok` is `False` and `dest` is already exists.
    """
    if os.path.exists(str(dest)):  # Wrapping in str is for Python 3.5
        message = "The configuration file {} already exists.".format(dest)
        if exist_ok:
            print(message)
        else:
            raise FileExistsError(message)
    else:
        print("Generate {}".format(dest), file=sys.stderr)
        shutil.copy(str(base), str(dest))


def create_list(dest, wav_dir, exist_ok=True):
    """Create an audio list file based on a template.

    This does not overwrite the existing one.

    Parameters
    ----------
    dest : str or path-like
        The path of the list file you are creating.
    wav_dir : str or path-like
        The path of the directory of audio files.
        The name of directory must be that of speaker.
    exist_ok : bool
        If `False`, this function throws
        `IOError` (Python 2.7) or `FileExistsError` (Python 3 or later)
        when `dest` is already created.

    Raises
    ------
    IOError (Python 2.7) or FileExistsError (Python 3 or later)
        If `exist_ok` is `False` and `dest` is already exists.
        You can catch both of them by:
        >>> except IOError:

    Notes
    -----
    List example of the speaker `SPEAKER`::

        SPEAKER/001
        SPEAKER/002
        SPEAKER/003

    when there is a audio directory of him/her named `SPEAKER` that contains:
        * 001.wav
        * 002.wav
        * 003.wav
        * other_file.txt (ignored)
    Note that the delimiter `/` turns to `\\` in Windows.
    """
    if os.path.exists(str(dest)):
        message = "The list file {} already exists.".format(dest)
        if exist_ok:
            print(message)
        else:
            raise FileExistsError(message)
    else:
        print("Generate {}".format(dest))
        speaker_label = os.path.basename(str(wav_dir))
        lines = (os.path.join(str(speaker_label), os.path.splitext(str(wav_file_name))[0])
                 for wav_file_name in os.listdir(str(wav_dir))
                 if os.path.splitext(str(wav_file_name))[1] == ".wav")
        with open(str(dest), "w") as file_handler:
            for line in sorted(lines):
                print(line, file=file_handler)


USES = ("train", "eval")
LIST_SUFFIXES = {
    use: "_" + use + ".list" for use in USES}

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
            for use in USES}
        for speaker_part, speaker_label in LABELS.items()}
    SPEAKER_CONF_FILES = {
        part: CONF_DIR / "speaker" / (label + ".yml")
        for part, label in LABELS.items()}
    PAIR_CONF_FILE = CONF_DIR / "pair" / (SOURCE_TARGET_PAIR + ".yml")
    SAMPLING_RATE = args["SAMPLING_RATE"]

    # The first False is dummy for alignment
    #   between indexes of `args_execute_steps` and arguments
    # pylint: disable=invalid-name
    execute_steps = [False] \
        + [args["--step{}".format(step_index)] for step_index in range(1, 4)]

    # Execute all steps if no options on steps are given
    # Keep index #0 False in case you create a hotbed for bugs.
    if not any(execute_steps):
        raise("Please specify steps with options")

    if execute_steps[1]:
        print("### 1. create initial list files ###")
        # create list files for both the speakers
        for use in USES:
            for part, speaker in LABELS.items():
                create_list(LIST_FILES[part][use], WAV_DIR / speaker)
        print("# Please modify train and eval list files, if you want. #")

    if execute_steps[2]:
        print("### 2. create configure files ###")
        # create speaker-dependent configure file
        for part, speaker in LABELS.items():
            create_configure(
                SPEAKER_CONF_FILES[part],
                CONF_DIR / "default" / "speaker_default_{}.yml".format(
                    SAMPLING_RATE))
        # create pair-dependent configure file
        create_configure(PAIR_CONF_FILE, os.path.join(
            str(CONF_DIR), "default", "pair_default.yml"))

    if execute_steps[3]:
        print("### 3. create figures to define F0 range ###")
        # get F0 range in each speaker
        for part, speaker in LABELS.items():
            initialize_speaker.main(
                speaker, str(LIST_FILES[part]["train"]),
                str(WAV_DIR), str(CONF_DIR / "figure"))
        print("# Please modify f0 range values"
              " in speaker-dependent YAML files based on the figure #")
