#!/bin/csh
sphinx-apidoc -f -o ./source ../sprocket
make html
