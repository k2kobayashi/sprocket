sprocket (vctk)
======

Voice conversion toolkit - Voice conversion (VC) is a technique to convert a speaker identity of a source speaker into that of a target speaker. In this framework, it enables us to develop a joint feature vector between source and target speech samples aligned using dynamic time warping (DTW) and model it based on statistical conversion models such as Gaussian mixture model (GMM), differential GMM (DIFFGMM), and deep neural networks (DNN).

[![Build Status](http://img.shields.io/travis/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}/master.svg)](https://travis-ci.org/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}})
[![Coverage Status](http://img.shields.io/coveralls/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}/master.svg)](https://coveralls.io/r/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}})
[![Scrutinizer Code Quality](http://img.shields.io/scrutinizer/g/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}.svg)](https://scrutinizer-ci.com/g/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}/?branch=master)
[![PyPI Version](http://img.shields.io/pypi/v/{{vctk}}.svg)](https://pypi.python.org/pypi/{{vctk}})
[![PyPI Downloads](http://img.shields.io/pypi/dm/{{vctk}}.svg)](https://pypi.python.org/pypi/{{vctk}})

## Requirements

- world
- numpy
- scipy
- pysptk
- h5py

## Installation

### Install WORLD for python

``` 
git clone https://github.com/jimsotelo/world.py
cd world.py
bash build_world.sh
python setup.py develop
```

- lib/world`ディレクトリに移動して、`sudo ./waf install` としてWORLDを`/usr/local/lib` にインストールする（※必須ではないが、こうしておくと楽）


- pythonから`import world` として、エラーがでなければOK

### Other packages

```
pip install -r requirements.txt
```

### Install vctk

```bash
python setup.py develop
```

## Run demonstration script

```
cd scripts
python vc_demo.py
```

## Tests

```bash
nosetests  -s -v
```

## KNOWN ISSUES

- Not work yet

## REPORTING BUGS

For any questions or issues please visit:

```
https://github.com/k2kobayashi/sprocket/issues
```

## COPYRIGHT

Copyright  2017 

Released under the MIT license

https://opensource.org/licenses/mit-license.php