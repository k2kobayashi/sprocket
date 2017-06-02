vctk
======

Voice conversion toolkit

[![Build Status](http://img.shields.io/travis/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}/master.svg)](https://travis-ci.org/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}})
[![Coverage Status](http://img.shields.io/coveralls/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}/master.svg)](https://coveralls.io/r/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}})
[![Scrutinizer Code Quality](http://img.shields.io/scrutinizer/g/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}.svg)](https://scrutinizer-ci.com/g/{{cookiecutter.github_username}}/{{cookiecutter.github_repo}}/?branch=master)
[![PyPI Version](http://img.shields.io/pypi/v/{{vctk}}.svg)](https://pypi.python.org/pypi/{{vctk}})
[![PyPI Downloads](http://img.shields.io/pypi/dm/{{vctk}}.svg)](https://pypi.python.org/pypi/{{vctk}})

## Requirements

適宜追加していく

- numpy
- scipy
- world
- sptk

## Installation

とりあえず、うちわ向けに、特記すべきことを書いておきます。

### WORLDのインストール

WORLDとそのpythonラッパーのインストール手順

- https://github.com/jimsotelo/world.py をcloneする
- `bash build_world.sh` として、WORLDをコンパイルする
- `lib/world`ディレクトリに移動して、`sudo ./waf install` としてWORLDを`/usr/local/lib` にインストールする（※必須ではないが、こうしておくと楽）
- `python setup.py devlop` で WORLDのpython wrapperインストール完了
- pythonから`import world` として、エラーがでなければOK

### SPTK のインストール

SPTKのpythonラッパーの手順は以下

- https://github.com/r9y9/SPTK をcloneして、READMEに書いてあるように、python wrapperをenableにしてインストール
- pythonから `import sptk` としてエラーが出なければOK

### vctk（仮）のインストール

```bash
python setup.py develop
```

## Tests

```bash
nosetests  -s -v
```

## 注意事項

- gomi.txt というファイル名は付けない。ファイル名はわかりやすくする。


NAME
----

Sprocket - a voice conversion framework developed by Python 2/3

DESCRIPTION
-----------

Voice conversion (VC) is a technique to convert a speaker individuality of a source speaker into that of a target speaker. In this framework, it enables us to develop a joint feature vector between source and target speakers, which is aligned based on dynamic time warping and model some statistical conversion models such as Gaussian mixture model (GMM), differential GMM (DIFFGMM), and deep neural networks (DNN).

-  [1] K. Kobayashi, T. Toda and S. Nakamura, "F0 transformation techniques for statistical voice conversion with direct waveform modification with spectral differential,窶抉roc. SLT, pp. 693-700, San Diego, USA, Dec. 2016.

USAGE
-----

- store waveform

  ``` Store waveform
cp $wav $twav
  ```

- Perform initialization and modify some paramet

``` Init
python init.py $org $tar
```

- Run train

``` train
python train.py
```

###  Configuration

- param1:
- param2: mixture of
- param3: hyper parameter
- param4:

INSTALLATION
------------

### REQUIREMENTS

- Linux or MAC
  - Python v2.7.12
      - numpy
      - scipy
      - Tensor flow


### MANUAL

Grab a copy of Sprocket:

    git clone git://github.com/k2kobayashi/sprocket.git

Install Sprocket with its requirements:

    python setup.py install

KNOWN ISSUES
------------

- Not work yes


REPORTING BUGS
--------------

For any questions or issues please visit:

    https://github.com/k2kobayashi/sprocket/issues

AUTHORS
-------

Sproket was originally written by K. KOBAYASHI and P. L. Tobing.

COPYRIGHT
---------

Copyright ﾂｩ 2017 Kazuhiro KOBAYASHI

Released under the MIT license

https://opensource.org/licenses/mit-license.php
