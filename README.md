[![Wercker](https://img.shields.io/wercker/ci/wercker/docs.svg)](https://app.wercker.com/k2kobayashi/sprocket)
[![PyPI Version](http://img.shields.io/pypi/v/{{sprocket}}.svg)](https://pypi.python.org/pypi/{{sprocket}})
[![PyPI Downloads](http://img.shields.io/pypi/dm/{{sproket}}.svg)](https://pypi.python.org/pypi/{{sprocket}})
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

sprocket
======

Voice conversion toolkit - Voice conversion (VC) is a technique to convert a speaker identity of a source speaker into that of a target speaker. In this framework, it enables for the users to develop a joint feature vector using parallel dataset between source and target speech samples and model it based on a Gaussian mixture model (GMM) and differential GMM (DIFFGMM).


## Purpose
### Reproduce VC based on the GMM and DIFFVC based on the DIFFGMM

In the major purpose of this framework, it enables for the users to implement voice conversion only preparing parallel dataset and excusing example scripts.
As the details of conversion method, please see the following papers.

- Toda et al., "Voice Conversion Based on Maximum-Likelihood Estimation of Spectral Parameter Trajectory," IEEE Trans. on ASLP, Vol. 15, No. 8, pp. 2222-2235, Nov. 2007
- Kobayashi et al., "F0 TRANSFORMATION TECHNIQUES FOR STATISTICAL VOICE CONVERSION WITH DIRECT WAVEFORM MODIFICATION WITH SPECTRAL DIFFERENTIAL," Proc. IEEE SLT, pp. 693-700, Dec 2016.

### Supply Python2/3 library related to VC
To easily develop VC and speech-based application using Python, the library of sprocket supplies several interfaces such ash acoustic feature analysis/synthesis, acoustic feature modeling and acoustic feature modifications.
For the details of the library, please see sprocket documents in https://hogehoge.hoge

## Installation & Run

### Current stable version

Ver. 0.20

### Install requirements

```
pip install numpy # for dependency
pip install -r requirements.txt
```

### Install sprocket

```
python setup.py develop
```

### Run example

See [Voice Conversion Example](docs/vc_example.md)

## REPORTING BUGS

For any questions or issues please visit:

```
https://github.com/k2kobayashi/sprocket/issues
```

## COPYRIGHT

Copyright (c) 2017 Kazuhiro KOBAYASHI

Released under the MIT license

https://opensource.org/licenses/mit-license.php

## ACKNOWLEDGEMENTS
Thank you, thank you.

