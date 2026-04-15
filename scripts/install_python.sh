#!/usr/bin/bash

# Install a given python version (default is 3.12) by compiling from source
# Credit: https://www.debugpoint.com/install-python-3-12-ubuntu/

# Notes:
#   1. Will install in current working directory
#   2. Will not install pip

local PYTHON_VERSION
PYTHON_VERSION=3.12.12

mkdir ~/src
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz
tar -zxvf Python-$PYTHON_VERSION.tgz
cd Python-$PYTHON_VERSION
mkdir ~/.localpython
./configure --prefix=$HOME/.localpython
make
make install
PATH=$HOME/.localpython:$PATH
cd -
rm Python-$PYTHON_VERSION.tgz
rm Python-$PYTHON_VERSION -rf
