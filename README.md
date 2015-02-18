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