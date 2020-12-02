[![Python Version](https://img.shields.io/badge/Python-3.5%2C%203.6%2C%203.7-green.svg)](https://img.shields.io/badge/Python-3.5%2C%203.6%2C%203.7-green.svg)
[![Build Status](https://www.travis-ci.org/k2kobayashi/sprocket.svg?branch=travis)](https://www.travis-ci.org/k2kobayashi/sprocket)
[![Coverage Status](https://coveralls.io/repos/github/k2kobayashi/sprocket/badge.svg?branch=master)](https://coveralls.io/github/k2kobayashi/sprocket?branch=master)
[![PyPI version](https://badge.fury.io/py/sprocket-vc.svg)](https://badge.fury.io/py/sprocket-vc)
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

sprocket
======


Voice conversion software - Voice conversion (VC) is a technique to convert a speaker identity of a source speaker into that of a target speaker. This software enables the users to develop a traditional VC system based on a Gaussian mixture model (GMM) and a vocoder-free VC system based on a differential GMM (DIFFGMM) using a parallel dataset of the source and target speakers.

## Paper and slide
- K. Kobayashi, T. Toda, "sprocket: Open-Source Voice Conversion Software," Proc. Odyssey, pp. 203-210, June 2018.
[[paper]](https://nuss.nagoya-u.ac.jp/s/h8YKnq6qxjjxtU3)

- T. Toda, "Hands on Voice Conversion," Speech Processing Courses in Crete (SPCC), July 2018.
[[slide]](https://www.slideshare.net/NU_I_TODALAB/hands-on-voice-conversion)

## Conversion samples
- Voice Conversion Challenge 2018
	- [HUB Task](https://nuss.nagoya-u.ac.jp/s/3F8dxTcdQdXir9s)
	- [SPOKE Task](https://nuss.nagoya-u.ac.jp/s/ixwxa6DxYa68y4N)

## Purpose
### Reproduce the typical VC systems

This software was developed to make it possible for the users to easily build the VC systems by only preparing a parallel dataset of the desired source and target speakers and executing example scripts.
The following VC methods were implemented as the typical VC methods.

#### Traditional VC method based on GMM
- T. Toda, A.W. Black, K. Tokuda, "Voice conversion based on maximum likelihood estimation of spectral parameter trajectory," IEEE Transactions on Audio, Speech and Language Processing, Vol. 15, No. 8, pp. 2222-2235, Nov. 2007.

#### Vocoder-free VC method based on DIFFGMM
- K. Kobayashi, T. Toda, S. Nakamura, "F0 transformation techniques for statistical voice conversion with direct waveform modification with spectral differential," Proc. IEEE SLT, pp. 693-700, Dec. 2016.

### Supply Python3 VC library
To make it possible to easily develop VC-based applications using Python (Python3), the VC library is also supplied, including several interfaces, such as acoustic feature analysis/synthesis, acoustic feature modeling, acoustic feature conversion, and waveform modification.
For the details of the VC library, please see sprocket documents in (coming soon).

## Installation & Run

Please use Python3.

### Current stable version

Ver. 0.18.4

### Install sprocket

```
pip install numpy==1.15.4 cython  # for dependency
pip install sprocket-vc
```

### Run example

See [VC example](docs/vc_example.md)

## REPORTING BUGS

For any questions or issues please visit:

```
https://github.com/k2kobayashi/sprocket/issues
```

## COPYRIGHT

Copyright (c) 2020 Kazuhiro KOBAYASHI

Released under the MIT license

[https://opensource.org/licenses/mit-license.php](https://opensource.org/licenses/mit-license.php)

## ACKNOWLEDGEMENTS
Thank you [@r9y9](https://github.com/r9y9) and [@tats-u](https://github.com/tats-u) for lots of contributions and encouragement helps before release.

## Who we are
- Kazuhiro Kobayashi [@k2kobayashi](https://github.com/k2kobayashi) [maintainer, design and development]

- [Tomoki Toda](https://sites.google.com/site/tomokitoda/) [advisor]
