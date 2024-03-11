#!/bin/bash

SRC_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )/../src" &> /dev/null && pwd )
cd $SRC_DIR

pip install -e subpackages/library
pip install -e subpackages/core
pip install -e frontend
